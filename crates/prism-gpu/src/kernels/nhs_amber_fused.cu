#include "../../../prism-nhs/src/cuda/prism_governor_measurement.cuh"
#include "../../../prism-nhs/src/cuda/prism_asc_decision.cuh"
//! NHS-AMBER Fused Mega Kernel
//!
//! Single-stream GPU pipeline combining:
//! - Full AMBER ff14SB physics (bonds, angles, dihedrals, LJ, electrostatics)
//! - Langevin thermostat with dynamic temperature protocols
//! - SHAKE/RATTLE hydrogen constraints
//! - Holographic exclusion field (negative space mapping)
//! - Water density inference from holographic negative
//! - Neuromorphic LIF continuous observation
//! - UV bias pump-probe perturbation
//! - Spike-triggered snapshot capture
//! - Warp matrix for atomic-precision alignment
//!
//! All fused into shared memory, no CPU round-trips.
//! Target: 100,000+ timesteps/second streaming.

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <assert.h>
#include "nhs_biprism_motion_ownership.cuh"
#include "nhs_amber_coordinate_geometry.cuh"

// Preserve the fatal condition and report the actual kernel/source location.
// Keep the hardware trap even when assertions are disabled by a build flag.
#define PRISM_PHYSICS_TRAP() do { \
    assert(false && "PRISM_PHYSICS_TRAP"); \
    asm volatile("trap;"); \
} while (0)

// Include excited state dynamics for true UV photophysics
#include "nhs_excited_state.cuh"

// Adaptive Cryo-Thermal Protocol for three-phase cryptic site detection
// Phase 1: CRYO BURST (80K, HIGH UV, global sweep)
// Phase 2: THERMAL RAMP (80K→300K, validation)
// Phase 3: FOCUSED DIG (300K, exploitation)
#include "prism_adaptive_protocol.cuh"

// Cryo-thermal detection physics (UV absorption → thermal signatures)
#include "cryo_thermal_detection.cuh"

// PRISM-TWIN v3.0: GPU-resident protocol state (shared with protocol_director.cu)
#include "protocol_state.cuh"

// Ultra-sensitive multi-modal neuromorphic detector
// Channels: thermal spike, gradient, melt wave, correlation
#include "sensitive_detector.cuh"

// Advanced UV-LIF coupling for direct UV → spike correlation
// Mechanisms: thermal wavefront, dewetting halo, cooperative enhancement
#include "uv_lif_coupling.cuh"

// ============================================================================
// CONSTANTS
// ============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_NEIGHBORS 128
#define MAX_GRID_DIM 128
#define PI 3.14159265358979323846f

// AMBER force field constants
#define COULOMB_CONSTANT 332.0636f  // kcal/mol * Angstrom / e^2
#define FUDGE_LJ 0.5f               // 1-4 LJ scaling
#define FUDGE_QQ 0.8333f            // 1-4 electrostatic scaling
#define AMBER_KCAL_TO_AMU_A2_PS2 418.4f
#define KB_KCAL_MOL_K 0.001987204f
#define MAX_INTEGRATOR_FORCE_KCAL_MOL_A 1000.0f

// Gate-5 anchor pin.  This magnitude is not a free tuning knob.  gearbox.cuh
// declares PRISM_GEAR_DEFAULT_FORCE_CLAMP_REFERENCE 1000.0f as "the
// integrator's own force clamp, MAX_INTEGRATOR_FORCE_KCAL_MOL_A" and expresses
// every GATE 5 force band as a fraction of it (0.25 OK / 0.50 HOLD / 1.00
// DOWNSHIFT / 4.00 EMERGENCY), and the Rust policy test
// gearbox_governor_policy.rs requires force_clamp_reference == 1000.0.  The
// value itself is the CA-restraint ceiling k=50 kcal/mol/A^2 * 20 A = 1000
// kcal/mol/A (amber_replica_parallel.cu MAX_FORCE).  Changing it here silently
// retunes gate 5.  static_assert emits no code, so this pin is PTX-neutral.
static_assert(MAX_INTEGRATOR_FORCE_KCAL_MOL_A == 1000.0f,
              "gearbox GATE 5 force bands anchor on this clamp; "
              "gearbox_governor_policy.rs requires force_clamp_reference == 1000.0");

#define MD_PHASE_FORCE_N 0
#define MD_PHASE_OU_HALF_KICK_N 1
#define MD_PHASE_RATTLE_PRE_DRIFT 2
#define MD_PHASE_DRIFT_SAVE_XN 3
#define MD_PHASE_SHAKE_POSITION 4
#define MD_PHASE_FORCE_NP1 5
#define MD_PHASE_KICK_NP1_OU_HALF 6
#define MD_PHASE_OU_HALF_N 10
#define MD_PHASE_RATTLE_AFTER_OU_N 11
#define MD_PHASE_KICK_N 12
#define MD_PHASE_KICK_NP1 13
#define MD_PHASE_RATTLE_BEFORE_OU_NP1 14
#define MD_PHASE_OU_HALF_NP1 15

// NHS constants
#define EXCLUSION_CUTOFF 8.0f       // Angstroms for exclusion field
#define WATER_DENSITY_BULK 0.0334f  // molecules/A^3
#define LIF_THRESHOLD 0.5f           // Tuned threshold for cryo-UV water density changes
#define LIF_RESET 0.0f              // Reset potential
#define REFRACTORY_STEPS 250       // 250 steps * 0.004ps (HMR) = 1.0ps refractory period
#define UV_WAVELENGTH 280.0f        // nm - aromatic absorption
#define UV_LIF_WINDOW 100           // Steps within which LIF spike is causally linked to prior UV (~0.4 ps at dt=0.004ps/HMR)
#define KCC_RING_STEPS 64           // Event-driven ring-buffer depth for KCC per residue

// KCC event record: stored only when causal activity occurs (c_t > 0)
struct KCCEvent {
    float dx;           // CA displacement x since previous step
    float dy;           // CA displacement y
    float dz;           // CA displacement z
    unsigned short c;   // causal count for this residue at this step
    unsigned short flags; // legacy: cold=1,warm=2; v2 marker=0x8000, heating=4,cooling=8,cold-return=16; endpoint phase, not UV event identity
    unsigned int step;  // absolute simulation timestep
};

// ============================================================================
// WARP-LEVEL PRIMITIVES (for fast reductions without shared memory)
// ============================================================================

// Warp shuffle reduction for sum (no __syncthreads needed!)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Warp shuffle reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Warp shuffle reduction for min
__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Fast reciprocal square root (use hardware rsqrt)
__device__ __forceinline__ float fast_rsqrt(float x) {
    return rsqrtf(x);
}

// Fast inverse (use hardware rcp)
__device__ __forceinline__ float fast_rcp(float x) {
    return __frcp_rn(x);
}

// ============================================================================
// CELL LIST CONSTANTS (O(N) Neighbor Lists)
// ============================================================================
// Cell size should be >= cutoff to ensure all neighbors are in adjacent cells
#define NB_CUTOFF 10.0f             // Nonbonded cutoff (Angstroms)
#define NB_CUTOFF_SQ 100.0f         // Cutoff squared
#define CELL_SIZE 10.0f             // Cell dimension (= cutoff)
#define CELL_SIZE_INV 0.1f          // 1.0 / CELL_SIZE
#define MAX_CELLS_PER_DIM 32        // Max cells per dimension (32^3 = 32768 cells max)
#define MAX_TOTAL_CELLS 32768       // MAX_CELLS_PER_DIM^3
#define MAX_ATOMS_PER_CELL 512      // Max atoms that can fit in one cell
#define NEIGHBOR_LIST_SIZE 1024     // Full per-atom neighbor list (with buffer)
#define NEIGHBOR_LIST_BUFFER 1.2f   // 20% buffer for list reuse between rebuilds

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Temperature protocol for cryogenic probing
struct TemperatureProtocol {
    float start_temp;       // Starting temperature (K)
    float end_temp;         // Ending temperature (K)
    int ramp_steps;         // Steps to ramp
    int hold_steps;         // Steps to hold at end
    int current_step;       // Current step in protocol

    __device__ float get_temperature() const {
        if (current_step < ramp_steps) {
            float t = (float)current_step / (float)ramp_steps;
            return start_temp + t * (end_temp - start_temp);
        }
        return end_temp;
    }
};

// Bond parameter
struct BondParam {
    int i, j;
    float r0, k;
};

// Angle parameter
struct AngleParam {
    int i, j, k;
    float theta0, force_k;
};

// Dihedral parameter
struct DihedralParam {
    int i, j, k, l;
    int periodicity;
    float phase, force_k;
};

// LJ parameter per atom
struct LJParam {
    float sigma, epsilon;
};

// Compact, stable-address dual-endpoint planes used for every admitted lambda
// slot.  Every plane carries both physical endpoints; a missing endpoint is
// represented explicitly by a zero presence mask.  Whether a missing internal
// term is a persistent dummy (one-sided insertion/deletion) or a true absent
// identity-state term (occupied-to-occupied transform) is resolved only from
// the immutable per-slot endpoint occupancy mask passed to the resident node.
struct LambdaDualAtomPlane {
    int slot;
    int present_a;
    int present_b;
    int reserved;
    float sigma_a, epsilon_a;
    float sigma_b, epsilon_b;
};

struct LambdaDualBondPlane {
    int i, j, slot, active_a, active_b, reserved;
    float r0_a, k_a;
    float r0_b, k_b;
};

struct LambdaDualAnglePlane {
    int i, j, k, slot;
    int active_a, active_b;
    float theta0_a, force_k_a;
    float theta0_b, force_k_b;
};

struct LambdaDualTorsionPlane {
    int i, j, k, l;
    int slot, periodicity;
    int active_a, active_b;
    int improper, reserved;
    float phase_a, force_k_a;
    float phase_b, force_k_b;
};

struct LambdaDualExceptionPlane {
    int i, j, slot;
    int present_a, present_b, reserved;
    float charge_product_a, sigma_a, epsilon_a;
    float charge_product_b, sigma_b, epsilon_b;
};

// Hydrogen cluster for SHAKE
struct HCluster {
    int central_atom;
    int hydrogen_atoms[3];  // -1 for unused
    float bond_lengths[3];
    int n_hydrogens;
    float inv_mass_central;
    float inv_mass_h;
};

// UV target (aromatic residue)
struct UVTarget {
    int residue_id;
    int atom_indices[16];   // Atoms in aromatic ring
    int n_atoms;
    float absorption_strength;  // Trp > Tyr > Phe
    // CANONICAL chromophore type ordering (MUST match Rust):
    //   0 = TRP (Tryptophan)
    //   1 = TYR (Tyrosine)
    //   2 = PHE (Phenylalanine)
    //   3 = S-S (Disulfide)
    int aromatic_type;
};

// Aromatic neighbor list for vibrational transfer
struct AromaticNeighbors {
    int atom_indices[64];       // Atoms within 5Å of aromatic
    int n_neighbors;
};

// Spike event for capture
struct SpikeEvent {
    int timestep;
    int voxel_idx;
    float3 position;
    float intensity;
    int nearby_residues[8];
    int n_residues;
    // Enhanced metadata for downstream docking/analysis
    int spike_source;           // 1=UV, 2=LIF
    float wavelength_nm;        // UV wavelength that triggered (0 for LIF)
    int aromatic_type;          // 0=TRP, 1=TYR, 2=PHE, 3=SS, -1=none/LIF
    int aromatic_residue_id;    // residue ID of closest excited aromatic (-1 for LIF)
    float water_density;        // local water density at spike voxel
    float vibrational_energy;   // energy deposited by UV excitation (0 for LIF)
    int n_nearby_excited;       // number of excited aromatics in range (pi-stacking)
    float wd_change;            // |water_density - water_density_prev| = |∂WD/∂t| for SDST energy_gradient
    unsigned int phase_bits;    // 10-bit CCNS phase angle (0-1023 → 0-2π) from ProtocolState
};

// Warp matrix entry - maps voxel to atoms
struct WarpEntry {
    int voxel_idx;
    int atom_indices[16];   // Contributing atoms
    float atom_weights[16]; // Distance-based weights
    int n_atoms;
};

// ============================================================================
// SHARED MEMORY LAYOUT
// ============================================================================

// Per-block shared memory for fused kernel
struct SharedMemory {
    // Atom data tile
    float3 positions[BLOCK_SIZE];
    float3 velocities[BLOCK_SIZE];
    float3 forces[BLOCK_SIZE];
    float charges[BLOCK_SIZE];
    float masses[BLOCK_SIZE];
    int atom_types[BLOCK_SIZE];

    // Exclusion field tile (for local grid region)
    float exclusion_tile[8][8][8];
    float water_density_tile[8][8][8];

    // LIF state tile
    float lif_potential[8][8][8];
    int spike_flags[8][8][8];

    // Reduction buffers
    float energy_buffer[BLOCK_SIZE];
    int spike_count;
};

// ============================================================================
// AMBER FORCE KERNELS
// ============================================================================

// Bond force: E = k(r - r0)^2
template<class GeoVec>
__device__ void compute_bond_force(
    const GeoVec& pi, const GeoVec& pj,
    float r0, float k,
    float3& fi, float3& fj,
    // M1.2.17 — per-atom PE accumulator + atom indices.  Pass nullptr
    // for pe_components to disable PE accumulation (legacy callers).
    double* pe_components,
    int bi, int bj
) {
    using GeoReal = typename BpNativeGeometryVector<GeoVec>::Real;
    GeoVec rij = bp_native_vec<GeoVec>(pj.x - pi.x, pj.y - pi.y, pj.z - pi.z);
    GeoReal r = bp_native_sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

    if (r < 1e-8f) return;

    GeoReal dr = r - r0;
    GeoReal force_mag = -2.0f * k * dr / r;

    GeoVec f = bp_native_vec<GeoVec>(force_mag * rij.x, force_mag * rij.y, force_mag * rij.z);
    fi.x -= f.x; fi.y -= f.y; fi.z -= f.z;
    fj.x += f.x; fj.y += f.y; fj.z += f.z;

    // M1.2.17 — accumulate V_b = k·(r − r_eq)² split half-half across
    // the two bonded atoms (so Σ pe_components = Σ_bonds V_b).
    // AMBER ff14SB convention: K_r already pre-folds the 1/2 into k.
    if (pe_components != nullptr) {
        const double v_bond_half = (double)k * (double)dr * (double)dr * 0.5;
        atomicAdd(pe_components + bi, v_bond_half);
        atomicAdd(pe_components + bj, v_bond_half);
    }
}

// Angle force: V = k(theta - theta0)^2
template<class GeoVec>
__device__ void compute_angle_force(
    const GeoVec& pi, const GeoVec& pj, const GeoVec& pk,
    float theta0, float k,
    float3& fi, float3& fj, float3& fk,
    // M1.2.17 — per-atom PE accumulator + 3 atom indices.
    double* pe_components,
    int ai, int aj, int ak
) {
    using GeoReal = typename BpNativeGeometryVector<GeoVec>::Real;
    GeoVec rij = bp_native_vec<GeoVec>(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
    GeoVec rkj = bp_native_vec<GeoVec>(pk.x - pj.x, pk.y - pj.y, pk.z - pj.z);

    GeoReal rij_len = bp_native_sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);
    GeoReal rkj_len = bp_native_sqrt(rkj.x*rkj.x + rkj.y*rkj.y + rkj.z*rkj.z);

    if (rij_len < 1e-8f || rkj_len < 1e-8f) return;

    GeoReal dot = rij.x*rkj.x + rij.y*rkj.y + rij.z*rkj.z;
    GeoReal cos_theta = dot / (rij_len * rkj_len);
    cos_theta = bp_native_max(-1.0f, bp_native_min(1.0f, cos_theta));

    GeoReal theta = bp_native_acos(cos_theta);
    GeoReal dtheta = theta - theta0;

    GeoReal sin_theta = bp_native_sqrt(1.0f - cos_theta*cos_theta);
    if (sin_theta < 1e-8f) sin_theta = 1e-8f;

    GeoReal force_mag = -2.0f * k * dtheta / sin_theta;

    // Gradient computation
    GeoReal inv_rij = 1.0f / rij_len;
    GeoReal inv_rkj = 1.0f / rkj_len;

    GeoVec di = bp_native_vec<GeoVec>(
        (rkj.x * inv_rkj - cos_theta * rij.x * inv_rij) * inv_rij,
        (rkj.y * inv_rkj - cos_theta * rij.y * inv_rij) * inv_rij,
        (rkj.z * inv_rkj - cos_theta * rij.z * inv_rij) * inv_rij
    );

    GeoVec dk = bp_native_vec<GeoVec>(
        (rij.x * inv_rij - cos_theta * rkj.x * inv_rkj) * inv_rkj,
        (rij.y * inv_rij - cos_theta * rkj.y * inv_rkj) * inv_rkj,
        (rij.z * inv_rij - cos_theta * rkj.z * inv_rkj) * inv_rkj
    );

    fi.x -= force_mag * di.x; fi.y -= force_mag * di.y; fi.z -= force_mag * di.z;
    fk.x -= force_mag * dk.x; fk.y -= force_mag * dk.y; fk.z -= force_mag * dk.z;
    fj.x += force_mag * (di.x + dk.x);
    fj.y += force_mag * (di.y + dk.y);
    fj.z += force_mag * (di.z + dk.z);

    // M1.2.17 — accumulate V_a = k·(θ − θ_eq)² split into thirds
    // across the three angle atoms (so Σ pe_components = Σ_angles V_a).
    if (pe_components != nullptr) {
        const double v_angle_third =
            (double)k * (double)dtheta * (double)dtheta * (1.0 / 3.0);
        atomicAdd(pe_components + ai, v_angle_third);
        atomicAdd(pe_components + aj, v_angle_third);
        atomicAdd(pe_components + ak, v_angle_third);
    }
}

// Read-only constructor observation of the actual uploaded harmonic fields.
// Each term owns a disjoint output row. The production helpers below are the
// same functions called by the ordinary MD force path; no alternate force law
// or trajectory/RNG mutation is introduced. PE split slots retain the helper's
// existing per-atom half/third partition and floating-point operations.
extern "C" __global__ void nhs_harmonic_parameter_observe_v1(
    const float3* positions, int n_atoms,
    const BondParam* bonds, int n_bonds,
    const AngleParam* angles, int n_angles,
    float* bond_force_components, double* bond_energy_splits,
    float* angle_force_components, double* angle_energy_splits,
    int* status
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) status[3] = 0x48524d31; // HRM1: actual kernel wrote this generation marker.
    if (tid < n_bonds) {
        const BondParam b = bonds[tid];
        if (b.i < 0 || b.i >= n_atoms || b.j < 0 || b.j >= n_atoms) {
            atomicOr(status, 1);
        } else {
            float3 fi = make_float3(0.0f, 0.0f, 0.0f);
            float3 fj = make_float3(0.0f, 0.0f, 0.0f);
            compute_bond_force(positions[b.i], positions[b.j], b.r0, b.k,
                fi, fj, bond_energy_splits + (size_t)tid * 2, 0, 1);
            const size_t base = (size_t)tid * 6;
            bond_force_components[base + 0] = fi.x;
            bond_force_components[base + 1] = fi.y;
            bond_force_components[base + 2] = fi.z;
            bond_force_components[base + 3] = fj.x;
            bond_force_components[base + 4] = fj.y;
            bond_force_components[base + 5] = fj.z;
            if (!isfinite(fi.x) || !isfinite(fi.y) || !isfinite(fi.z)
                || !isfinite(fj.x) || !isfinite(fj.y) || !isfinite(fj.z)
                || !isfinite(bond_energy_splits[(size_t)tid * 2])
                || !isfinite(bond_energy_splits[(size_t)tid * 2 + 1])) atomicOr(status, 4);
        }
        atomicAdd(status + 1, 1);
    }
    if (tid < n_angles) {
        const AngleParam a = angles[tid];
        if (a.i < 0 || a.i >= n_atoms || a.j < 0 || a.j >= n_atoms
            || a.k < 0 || a.k >= n_atoms) {
            atomicOr(status, 2);
        } else {
            float3 fi = make_float3(0.0f, 0.0f, 0.0f);
            float3 fj = make_float3(0.0f, 0.0f, 0.0f);
            float3 fk = make_float3(0.0f, 0.0f, 0.0f);
            compute_angle_force(positions[a.i], positions[a.j], positions[a.k],
                a.theta0, a.force_k, fi, fj, fk,
                angle_energy_splits + (size_t)tid * 3, 0, 1, 2);
            const size_t base = (size_t)tid * 9;
            angle_force_components[base + 0] = fi.x;
            angle_force_components[base + 1] = fi.y;
            angle_force_components[base + 2] = fi.z;
            angle_force_components[base + 3] = fj.x;
            angle_force_components[base + 4] = fj.y;
            angle_force_components[base + 5] = fj.z;
            angle_force_components[base + 6] = fk.x;
            angle_force_components[base + 7] = fk.y;
            angle_force_components[base + 8] = fk.z;
            if (!isfinite(fi.x) || !isfinite(fi.y) || !isfinite(fi.z)
                || !isfinite(fj.x) || !isfinite(fj.y) || !isfinite(fj.z)
                || !isfinite(fk.x) || !isfinite(fk.y) || !isfinite(fk.z)
                || !isfinite(angle_energy_splits[(size_t)tid * 3])
                || !isfinite(angle_energy_splits[(size_t)tid * 3 + 1])
                || !isfinite(angle_energy_splits[(size_t)tid * 3 + 2])) atomicOr(status, 8);
        }
        atomicAdd(status + 2, 1);
    }
}

// Dihedral force: V = (V_n/2)·[1 + cos(n*phi - gamma)]
// (AMBER convention: k stored = V_n/2 ⇒ V = k·(1 + cos(n·phi − gamma))).
template<class GeoVec>
__device__ void compute_dihedral_force(
    const GeoVec& pi, const GeoVec& pj, const GeoVec& pk, const GeoVec& pl,
    int periodicity, float phase, float k,
    float3& fi, float3& fj, float3& fk, float3& fl,
    // M1.2.17 — per-atom PE accumulator + 4 atom indices.
    double* pe_components,
    int di, int dj, int dk, int dl
) {
    using GeoReal = typename BpNativeGeometryVector<GeoVec>::Real;
    // Match OpenMM's PeriodicTorsionForce atom ordering and analytic gradient:
    // v0 = r1-r2, v1 = r3-r2, v2 = r3-r4.  This distributes the force to
    // all four atoms with the complete chain rule, including the central-atom
    // projection terms.  Do not replace this with endpoint splitting.
    const GeoVec v0 = bp_native_vec<GeoVec>(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
    const GeoVec v1 = bp_native_vec<GeoVec>(pk.x - pj.x, pk.y - pj.y, pk.z - pj.z);
    const GeoVec v2 = bp_native_vec<GeoVec>(pk.x - pl.x, pk.y - pl.y, pk.z - pl.z);

    const GeoVec cp0 = bp_native_vec<GeoVec>(
        v0.y*v1.z - v0.z*v1.y,
        v0.z*v1.x - v0.x*v1.z,
        v0.x*v1.y - v0.y*v1.x
    );
    const GeoVec cp1 = bp_native_vec<GeoVec>(
        v1.y*v2.z - v1.z*v2.y,
        v1.z*v2.x - v1.x*v2.z,
        v1.x*v2.y - v1.y*v2.x
    );

    const GeoReal norm_cross1 = cp0.x*cp0.x + cp0.y*cp0.y + cp0.z*cp0.z;
    const GeoReal norm_cross2 = cp1.x*cp1.x + cp1.y*cp1.y + cp1.z*cp1.z;
    const GeoReal norm_sqr_bc = v1.x*v1.x + v1.y*v1.y + v1.z*v1.z;
    if (norm_cross1 < 1e-16f || norm_cross2 < 1e-16f || norm_sqr_bc < 1e-16f) return;

    const GeoReal norm_bc = bp_native_sqrt(norm_sqr_bc);
    GeoReal cos_phi = (cp0.x*cp1.x + cp0.y*cp1.y + cp0.z*cp1.z) /
                    bp_native_sqrt(norm_cross1 * norm_cross2);
    cos_phi = bp_native_max(-1.0f, bp_native_min(1.0f, cos_phi));

    GeoReal phi;
    if (cos_phi > 0.99f || cos_phi < -0.99f) {
        const GeoVec cp_cross = bp_native_vec<GeoVec>(
            cp0.y*cp1.z - cp0.z*cp1.y,
            cp0.z*cp1.x - cp0.x*cp1.z,
            cp0.x*cp1.y - cp0.y*cp1.x
        );
        GeoReal sin_sqr = (cp_cross.x*cp_cross.x + cp_cross.y*cp_cross.y +
                         cp_cross.z*cp_cross.z) / (norm_cross1 * norm_cross2);
        sin_sqr = bp_native_max(0.0f, bp_native_min(1.0f, sin_sqr));
        phi = bp_native_asin(bp_native_sqrt(sin_sqr));
        if (cos_phi < 0.0f) phi = PI - phi;
    } else {
        phi = bp_native_acos(cos_phi);
    }
    const GeoReal sign_probe = v0.x*cp1.x + v0.y*cp1.y + v0.z*cp1.z;
    if (sign_probe < 0.0f) phi = -phi;

    const GeoReal n = (float)periodicity;
    const GeoReal dE_dphi = -k * n * bp_native_sin(n * phi - phase);
    const GeoReal inv_norm_sqr_bc = 1.0f / norm_sqr_bc;
    const GeoReal ff0 = (-dE_dphi * norm_bc) / norm_cross1;
    const GeoReal ff1 = (v0.x*v1.x + v0.y*v1.y + v0.z*v1.z) * inv_norm_sqr_bc;
    const GeoReal ff2 = (v2.x*v1.x + v2.y*v1.y + v2.z*v1.z) * inv_norm_sqr_bc;
    const GeoReal ff3 = (dE_dphi * norm_bc) / norm_cross2;

    const GeoVec force1 = bp_native_vec<GeoVec>(ff0*cp0.x, ff0*cp0.y, ff0*cp0.z);
    const GeoVec force4 = bp_native_vec<GeoVec>(ff3*cp1.x, ff3*cp1.y, ff3*cp1.z);
    const GeoVec s = bp_native_vec<GeoVec>(
        ff1*force1.x - ff2*force4.x,
        ff1*force1.y - ff2*force4.y,
        ff1*force1.z - ff2*force4.z
    );
    const GeoVec force2 = bp_native_vec<GeoVec>(
        s.x - force1.x,
        s.y - force1.y,
        s.z - force1.z
    );
    const GeoVec force3 = bp_native_vec<GeoVec>(
        -s.x - force4.x,
        -s.y - force4.y,
        -s.z - force4.z
    );

    fi.x += force1.x; fi.y += force1.y; fi.z += force1.z;
    fj.x += force2.x; fj.y += force2.y; fj.z += force2.z;
    fk.x += force3.x; fk.y += force3.y; fk.z += force3.z;
    fl.x += force4.x; fl.y += force4.y; fl.z += force4.z;

    // M1.2.17 — accumulate V_d = k·(1 + cos(n·phi − gamma)) split into
    // quarters across the four dihedral atoms.  Note: V_n/2 is pre-folded
    // into k in the AMBER topology, so V_d = k·(1 + cos(n·phi − gamma)).
    // Σ pe_components = Σ_dihedrals V_d.
    if (pe_components != nullptr) {
        const double v_dih = (double)k * (1.0 + cos((double)n * (double)phi - (double)phase));
        const double v_dih_quarter = v_dih * 0.25;
        atomicAdd(pe_components + di, v_dih_quarter);
        atomicAdd(pe_components + dj, v_dih_quarter);
        atomicAdd(pe_components + dk, v_dih_quarter);
        atomicAdd(pe_components + dl, v_dih_quarter);
    }
}

// Ordinary cutoff forces have the continuous piecewise potential
// U(r)-U(rc) for r<rc, zero outside. The old unshifted energy introduced
// finite charge-dependent jumps when pairs entered or left the neighbor domain.
// This scalar offset is position-independent INSIDE the domain: every existing
// interior force is retained. Exact exception pairs are never shifted.
template<class GeometryReal=float>
__device__ __forceinline__ double ordinary_cutoff_potential_offset_v1(
    float sigma,float epsilon,float charge_product,float cutoff_sq,
    float scale_lj,float scale_qq
) {
    const GeometryReal inv_rc=bp_native_rsqrt((GeometryReal)cutoff_sq);
    const GeometryReal sr=sigma*inv_rc,sr2=sr*sr,sr6=sr2*sr2*sr2,sr12=sr6*sr6;
    return (double)scale_lj*4.0*(double)epsilon*((double)sr12-(double)sr6)
        +(double)scale_qq*(double)COULOMB_CONSTANT*(double)charge_product*(double)inv_rc;
}

// LJ + Electrostatic nonbonded force (OPTIMIZED with fast math)
template<class GeoVec>
__device__ __forceinline__ void compute_nonbonded_force(
    const GeoVec& pi, const GeoVec& pj,
    float qi, float qj,
    float sigma_i, float epsilon_i,
    float sigma_j, float epsilon_j,
    float3& fi, float3& fj,
    float cutoff_sq,
    // M1.2.17 — per-atom PE accumulator + 2 atom indices.  pe_components
    // = nullptr disables PE accumulation (legacy callers).
    double* pe_components,
    int nbi, int nbj,
    // Legacy 1-4 pairs use ff14SB FUDGE values. Exact exception pairs use
    // the prmtop/OpenMM charge product and LJ values passed alongside them.
    bool is_14,
    bool exact_14,
    float exact_charge_product,
    float exact_sigma,
    float exact_epsilon,
    bool historical_may_model = false
) {
    using GeoReal = typename BpNativeGeometryVector<GeoVec>::Real;
    GeoVec rij = bp_native_vec<GeoVec>(pj.x - pi.x, pj.y - pi.y, pj.z - pi.z);
    GeoReal r_sq = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;

    // OpenMM NonbondedForce exceptions are explicit bonded corrections and
    // remain active outside the ordinary nonbonded cutoff.
    if ((!exact_14 && r_sq > cutoff_sq) || r_sq < 1e-8f) return;

    // OPTIMIZATION: Use rsqrtf (hardware intrinsic) instead of sqrt + divide
    GeoReal inv_r = bp_native_rsqrt(r_sq);  // 1/sqrt(r_sq) in single instruction
    GeoReal inv_r2 = inv_r * inv_r;

    float sigma = exact_14 ? exact_sigma : 0.5f * (sigma_i + sigma_j);
    float epsilon = exact_14 ? exact_epsilon : sqrtf(epsilon_i * epsilon_j);
    float charge_product = exact_14 ? exact_charge_product : qi * qj;

    // LJ force - precompute powers efficiently
    GeoReal sigma_r = sigma * inv_r;
    GeoReal sigma_r2 = sigma_r * sigma_r;
    GeoReal sigma_r6 = sigma_r2 * sigma_r2 * sigma_r2;  // (sigma/r)^6
    GeoReal sigma_r12 = sigma_r6 * sigma_r6;
    GeoReal lj_force = 24.0f * epsilon * inv_r2 * (2.0f * sigma_r12 - sigma_r6);

    // Coulomb force
    GeoReal elec_force = COULOMB_CONSTANT * charge_product * inv_r2 * inv_r;

    // M1.2.18-P2.3 — 1-4 FUDGE applied to BOTH force and PE.
    // Branchless select.
    const float scale_lj = (is_14 && !exact_14) ? FUDGE_LJ : 1.0f;
    const float scale_qq = (is_14 && !exact_14) ? FUDGE_QQ : 1.0f;
    GeoReal total_force = scale_lj * lj_force + scale_qq * elec_force;

    GeoVec f = bp_native_vec<GeoVec>(total_force * rij.x, total_force * rij.y, total_force * rij.z);
    fi.x -= f.x; fi.y -= f.y; fi.z -= f.z;
    fj.x += f.x; fj.y += f.y; fj.z += f.z;

    // M1.2.17 — accumulate V_pair = V_LJ + V_elec (with 1-4 FUDGE
    // applied per M1.2.18-P2.3) split half-half across atoms i, j.
    //   V_LJ   = 4·ε·(σ_r¹² − σ_r⁶)         (Lorentz-Berthelot ε,σ form)
    //   V_elec = COULOMB_CONSTANT·q_i·q_j / r
    if (pe_components != nullptr) {
        const double v_lj   = 4.0 * (double)epsilon
                                * ((double)sigma_r12 - (double)sigma_r6);
        const double v_elec = (double)COULOMB_CONSTANT * (double)charge_product
                                * (double)inv_r;
        const double v_pair = (double)scale_lj * v_lj
                            + (double)scale_qq * v_elec
                            - ((exact_14 || historical_may_model) ? 0.0 : ordinary_cutoff_potential_offset_v1<GeoReal>(
                                sigma,epsilon,charge_product,cutoff_sq,scale_lj,scale_qq));
        const double v_pair_half = v_pair * 0.5;
        atomicAdd(pe_components + nbi, v_pair_half);
        atomicAdd(pe_components + nbj, v_pair_half);
    }
}

// Legacy direct-space V1 lambda-fragment soft-core LJ+Coulomb reference.
// Production PME execution uses the distinct V2 functions below; these two
// Hamiltonians must never be mixed in one trajectory.
template<class GeoVec>
__device__ __forceinline__ void compute_lambda_softcore_pair(
    const GeoVec& pi, const GeoVec& pj,
    float charge_product, float sigma, float epsilon, float lambda_pair,
    float alpha_lj, float alpha_q_a2, float softcore_power, float coupling_power,
    float cutoff_sq, bool exact_pair,
    float3& force_i, double& potential, double& du_dlambda_pair
) {
    using GeoReal = typename BpNativeGeometryVector<GeoVec>::Real;
    const GeoReal dx = pi.x - pj.x;
    const GeoReal dy = pi.y - pj.y;
    const GeoReal dz = pi.z - pj.z;
    const GeoReal r2 = dx*dx + dy*dy + dz*dz;
    if ((!exact_pair && r2 > cutoff_sq) || r2 < 1.0e-8f) return;
    const double r2d = (double)r2;
    const double r = sqrt(r2d);
    const double l = (double)lambda_pair;
    const double one_minus = 1.0 - l;
    const double scale = pow(l, (double)coupling_power);
    const double dscale = l == 0.0
        ? ((double)coupling_power == 1.0 ? 1.0 : 0.0)
        : (double)coupling_power * pow(l, (double)coupling_power - 1.0);
    const double soften = pow(one_minus, (double)softcore_power);
    const double dsoften = one_minus == 0.0
        ? ((double)softcore_power == 1.0 ? -1.0 : 0.0)
        : -(double)softcore_power * pow(one_minus, (double)softcore_power - 1.0);

    const double sigma2 = (double)sigma * (double)sigma;
    const double sigma6 = sigma2 * sigma2 * sigma2;
    const double r6 = r2d * r2d * r2d;
    const double sc6 = r6 + (double)alpha_lj * soften * sigma6;
    const double x = sigma6 / sc6;
    const double lj_shape = x*x - x;
    const double lj = 4.0 * (double)epsilon * scale * lj_shape;
    const double dsc6_dl = (double)alpha_lj * sigma6 * dsoften;
    const double dx_dl = -sigma6 * dsc6_dl / (sc6*sc6);
    const double dlj_dl = 4.0 * (double)epsilon
        * (dscale*lj_shape + scale*(2.0*x - 1.0)*dx_dl);
    const double dx_dr = -sigma6 * (6.0 * pow(r, 5.0)) / (sc6*sc6);
    const double dlj_dr = 4.0 * (double)epsilon * scale * (2.0*x - 1.0) * dx_dr;

    const double sc2 = r2d + (double)alpha_q_a2 * soften;
    const double inv_sc = 1.0 / sqrt(sc2);
    const double qpref = (double)COULOMB_CONSTANT * (double)charge_product;
    const double q = qpref * scale * inv_sc;
    const double dsc2_dl = (double)alpha_q_a2 * dsoften;
    const double dinv_sc_dl = -0.5 * dsc2_dl * inv_sc*inv_sc*inv_sc;
    const double dq_dl = qpref * (dscale*inv_sc + scale*dinv_sc_dl);
    const double dq_dr = -qpref * scale * r * inv_sc*inv_sc*inv_sc;

    const double radial_force = -(dlj_dr + dq_dr);
    const double inv_r = 1.0 / r;
    force_i.x += (float)(radial_force * (double)dx * inv_r);
    force_i.y += (float)(radial_force * (double)dy * inv_r);
    force_i.z += (float)(radial_force * (double)dz * inv_r);
    potential += lj + q;
    du_dlambda_pair += dlj_dl + dq_dl;
}

// Production V2 soft-core LJ component. Electrostatics are charge-linear
// Ewald/PME and are evaluated by the Ewald functions below.
template<class GeoVec>
__device__ __forceinline__ void compute_lambda_softcore_lj_v2_pair(
    const GeoVec& pi, const GeoVec& pj,
    float sigma, float epsilon, float lambda_pair,
    float alpha_lj, float softcore_power, float coupling_power,
    float cutoff_sq, bool exact_pair,
    float3& force_i, double& potential, double& du_dlambda_pair
) {
    using GeoReal = typename BpNativeGeometryVector<GeoVec>::Real;
    const GeoReal dx = pi.x - pj.x;
    const GeoReal dy = pi.y - pj.y;
    const GeoReal dz = pi.z - pj.z;
    const double r2 = (double)dx*dx + (double)dy*dy + (double)dz*dz;
    if ((!exact_pair && r2 > (double)cutoff_sq) || r2 < 1.0e-16) return;
    const double r = sqrt(r2);
    const double l = (double)lambda_pair;
    const double one_minus = 1.0 - l;
    const double scale = pow(l, (double)coupling_power);
    const double dscale = l == 0.0
        ? ((double)coupling_power == 1.0 ? 1.0 : 0.0)
        : (double)coupling_power * pow(l, (double)coupling_power - 1.0);
    const double soften = pow(one_minus, (double)softcore_power);
    const double dsoften = one_minus == 0.0
        ? ((double)softcore_power == 1.0 ? -1.0 : 0.0)
        : -(double)softcore_power * pow(one_minus, (double)softcore_power - 1.0);
    const double sigma2 = (double)sigma * (double)sigma;
    const double sigma6 = sigma2 * sigma2 * sigma2;
    const double r6 = r2 * r2 * r2;
    const double sc6 = r6 + (double)alpha_lj * soften * sigma6;
    const double x = sigma6 / sc6;
    const double shape = x*x - x;
    const double energy = 4.0 * (double)epsilon * scale * shape;
    const double dsc6_dl = (double)alpha_lj * sigma6 * dsoften;
    const double dx_dl = -sigma6 * dsc6_dl / (sc6*sc6);
    const double dudl = 4.0 * (double)epsilon
        * (dscale*shape + scale*(2.0*x - 1.0)*dx_dl);
    const double dx_dr = -sigma6 * (6.0 * pow(r, 5.0)) / (sc6*sc6);
    const double radial_force = -4.0 * (double)epsilon * scale * (2.0*x - 1.0) * dx_dr;
    const double inv_r = 1.0 / r;
    force_i.x += (float)(radial_force * (double)dx * inv_r);
    force_i.y += (float)(radial_force * (double)dy * inv_r);
    force_i.z += (float)(radial_force * (double)dz * inv_r);
    potential += energy;
    du_dlambda_pair += dudl;
}

extern "C" __global__ void nhs_lambda_fragment_softcore_lj_reference_v2(
    float dx, float dy, float dz,
    float lambda_pair, float sigma, float epsilon,
    float alpha_lj, float softcore_power, float coupling_power, float cutoff,
    double* potential_out, float* force_out, double* dudl_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float3 pi = make_float3(dx, dy, dz);
    const float3 pj = make_float3(0.0f, 0.0f, 0.0f);
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    double potential = 0.0;
    double dudl = 0.0;
    compute_lambda_softcore_lj_v2_pair(
        pi, pj, sigma, epsilon, lambda_pair,
        alpha_lj, softcore_power, coupling_power,
        cutoff*cutoff, false, force, potential, dudl);
    potential_out[0] = potential;
    force_out[0] = force.x;
    force_out[1] = force.y;
    force_out[2] = force.z;
    dudl_out[0] = dudl;
}

template<class GeoVec>
__device__ __forceinline__ GeoVec minimum_image_orthorhombic_v1(
    GeoVec displacement, float3 box_lengths
) {
    displacement.x -= box_lengths.x * bp_native_nearbyint(displacement.x / box_lengths.x);
    displacement.y -= box_lengths.y * bp_native_nearbyint(displacement.y / box_lengths.y);
    displacement.z -= box_lengths.z * bp_native_nearbyint(displacement.z / box_lengths.z);
    return displacement;
}

__device__ __forceinline__ void lambda_atomic_add_force(
    float3* forces, int atom, const float3& value
) {
    atomicAdd(&forces[atom].x, value.x);
    atomicAdd(&forces[atom].y, value.y);
    atomicAdd(&forces[atom].z, value.z);
}

__device__ __forceinline__ bool lambda_csr_contains(
    int owner, int partner, const int* values, const int* offsets
) {
    for (int index = offsets[owner]; index < offsets[owner + 1]; ++index) {
        if (values[index] == partner) return true;
    }
    return false;
}

// One dual-endpoint LJ path. Common physical atoms use exact endpoint-energy
// interpolation. Appearing/disappearing union atoms use the production V2
// soft-core leg in the correct direction. The returned force and dU/dlambda
// therefore derive from the exact same endpoint Hamiltonian expression.
template<class GeoVec>
__device__ __forceinline__ void compute_lambda_dual_endpoint_lj_v3_pair(
    GeoVec displacement, float lambda,
    int present_a, float sigma_a, float epsilon_a,
    int present_b, float sigma_b, float epsilon_b,
    float alpha_lj, float softcore_power, float coupling_power,
    float cutoff_sq, bool exact_pair,
    float3& force_i, double& potential, double& du_dlambda
) {
    if (present_a == 0 && present_b == 0) PRISM_PHYSICS_TRAP();
    const GeoVec zero = bp_native_vec<GeoVec>(0.0f, 0.0f, 0.0f);
    float3 force_a = make_float3(0.0f, 0.0f, 0.0f);
    float3 force_b = make_float3(0.0f, 0.0f, 0.0f);
    double energy_a = 0.0;
    double energy_b = 0.0;
    double endpoint_dudl = 0.0;
    if (present_a != 0) {
        compute_lambda_softcore_lj_v2_pair(
            displacement, zero, sigma_a, epsilon_a, 1.0f,
            alpha_lj, softcore_power, coupling_power,
            cutoff_sq, exact_pair, force_a, energy_a, endpoint_dudl);
    }
    endpoint_dudl = 0.0;
    if (present_b != 0) {
        compute_lambda_softcore_lj_v2_pair(
            displacement, zero, sigma_b, epsilon_b, 1.0f,
            alpha_lj, softcore_power, coupling_power,
            cutoff_sq, exact_pair, force_b, energy_b, endpoint_dudl);
    }
    if (present_a != 0 && present_b != 0) {
        const double l = (double)lambda;
        force_i.x += (float)((1.0-l)*(double)force_a.x + l*(double)force_b.x);
        force_i.y += (float)((1.0-l)*(double)force_a.y + l*(double)force_b.y);
        force_i.z += (float)((1.0-l)*(double)force_a.z + l*(double)force_b.z);
        potential += (1.0-l)*energy_a + l*energy_b;
        du_dlambda += energy_b - energy_a;
    } else if (present_b != 0) {
        compute_lambda_softcore_lj_v2_pair(
            displacement, zero, sigma_b, epsilon_b, lambda,
            alpha_lj, softcore_power, coupling_power,
            cutoff_sq, exact_pair, force_i, potential, du_dlambda);
    } else {
        double reverse_dudl = 0.0;
        compute_lambda_softcore_lj_v2_pair(
            displacement, zero, sigma_a, epsilon_a, 1.0f-lambda,
            alpha_lj, softcore_power, coupling_power,
            cutoff_sq, exact_pair, force_i, potential, reverse_dudl);
        du_dlambda -= reverse_dudl;
    }
}

// General cross-slot endpoint-aware LJ path. Occupancy coupling and chemical
// identity interpolation are distinct coordinates: c(lambda) is one for an
// occupied-to-occupied transform and ramps only for a one-sided slot, while
// identity weights interpolate the admitted A/B parameter planes. Every
// endpoint combination is evaluated under the same soft-core occupancy
// coupling g=c_i*c_j. This remains energy/force/derivative consistent when
// both slots are partial and reduces to ordinary lambda_i*lambda_j soft-core
// coupling for two EMPTY(A)->OCCUPIED(B) slots.
template<class GeoVec>
__device__ __forceinline__ void compute_lambda_dual_cross_slot_lj_v4_pair(
    GeoVec displacement, float lambda_i,
    int present_ia, float sigma_ia, float epsilon_ia,
    int present_ib, float sigma_ib, float epsilon_ib,
    float lambda_j,
    int present_ja, float sigma_ja, float epsilon_ja,
    int present_jb, float sigma_jb, float epsilon_jb,
    float alpha_lj, float softcore_power, float coupling_power,
    float cutoff_sq, bool exact_pair,
    float3& force_i, double& potential,
    double& du_dlambda_i, double& du_dlambda_j
) {
    if ((present_ia == 0 && present_ib == 0)
        || (present_ja == 0 && present_jb == 0)) PRISM_PHYSICS_TRAP();
    const bool identity_i = present_ia != 0 && present_ib != 0;
    const bool identity_j = present_ja != 0 && present_jb != 0;
    const double ci = identity_i ? 1.0
        : (present_ib != 0 ? (double)lambda_i : 1.0-(double)lambda_i);
    const double cj = identity_j ? 1.0
        : (present_jb != 0 ? (double)lambda_j : 1.0-(double)lambda_j);
    const double dci = identity_i ? 0.0 : (present_ib != 0 ? 1.0 : -1.0);
    const double dcj = identity_j ? 0.0 : (present_jb != 0 ? 1.0 : -1.0);
    const float occupancy_coupling = (float)(ci*cj);
    if (!isfinite(occupancy_coupling) || occupancy_coupling < 0.0f
        || occupancy_coupling > 1.0f) PRISM_PHYSICS_TRAP();

    const double wi[2] = {
        identity_i ? 1.0-(double)lambda_i : (present_ia != 0 ? 1.0 : 0.0),
        identity_i ? (double)lambda_i : (present_ib != 0 ? 1.0 : 0.0)
    };
    const double dwi[2] = {identity_i ? -1.0 : 0.0, identity_i ? 1.0 : 0.0};
    const double wj[2] = {
        identity_j ? 1.0-(double)lambda_j : (present_ja != 0 ? 1.0 : 0.0),
        identity_j ? (double)lambda_j : (present_jb != 0 ? 1.0 : 0.0)
    };
    const double dwj[2] = {identity_j ? -1.0 : 0.0, identity_j ? 1.0 : 0.0};
    const float sigma_i[2] = {sigma_ia, sigma_ib};
    const float epsilon_i[2] = {epsilon_ia, epsilon_ib};
    const float sigma_j[2] = {sigma_ja, sigma_jb};
    const float epsilon_j[2] = {epsilon_ja, epsilon_jb};
    const int present_i[2] = {present_ia, present_ib};
    const int present_j[2] = {present_ja, present_jb};
    const GeoVec zero = bp_native_vec<GeoVec>(0.0f, 0.0f, 0.0f);
    for (int a = 0; a < 2; ++a) {
        if (present_i[a] == 0 || wi[a] == 0.0) continue;
        for (int b = 0; b < 2; ++b) {
            if (present_j[b] == 0 || wj[b] == 0.0) continue;
            float3 component_force = make_float3(0.0f, 0.0f, 0.0f);
            double component_energy = 0.0;
            double component_dudg = 0.0;
            compute_lambda_softcore_lj_v2_pair(
                displacement, zero,
                0.5f*(sigma_i[a]+sigma_j[b]),
                sqrtf(epsilon_i[a]*epsilon_j[b]),
                occupancy_coupling,
                alpha_lj, softcore_power, coupling_power,
                cutoff_sq, exact_pair,
                component_force, component_energy, component_dudg);
            const double identity_weight = wi[a]*wj[b];
            force_i.x += (float)(identity_weight*(double)component_force.x);
            force_i.y += (float)(identity_weight*(double)component_force.y);
            force_i.z += (float)(identity_weight*(double)component_force.z);
            potential += identity_weight*component_energy;
            du_dlambda_i += dwi[a]*wj[b]*component_energy
                + identity_weight*component_dudg*dci*cj;
            du_dlambda_j += wi[a]*dwj[b]*component_energy
                + identity_weight*component_dudg*ci*dcj;
        }
    }
}

template<class GeoVec>
__device__ __forceinline__ typename BpNativeGeometryVector<GeoVec>::Real lambda_torsion_phi_v1(
    const GeoVec& pi, const GeoVec& pj, const GeoVec& pk, const GeoVec& pl
) {
    using GeoReal = typename BpNativeGeometryVector<GeoVec>::Real;
    const GeoVec v0 = bp_native_vec<GeoVec>(pi.x-pj.x, pi.y-pj.y, pi.z-pj.z);
    const GeoVec v1 = bp_native_vec<GeoVec>(pk.x-pj.x, pk.y-pj.y, pk.z-pj.z);
    const GeoVec v2 = bp_native_vec<GeoVec>(pk.x-pl.x, pk.y-pl.y, pk.z-pl.z);
    const GeoVec cp0 = bp_native_vec<GeoVec>(
        v0.y*v1.z-v0.z*v1.y, v0.z*v1.x-v0.x*v1.z, v0.x*v1.y-v0.y*v1.x);
    const GeoVec cp1 = bp_native_vec<GeoVec>(
        v1.y*v2.z-v1.z*v2.y, v1.z*v2.x-v1.x*v2.z, v1.x*v2.y-v1.y*v2.x);
    const GeoReal n0 = cp0.x*cp0.x+cp0.y*cp0.y+cp0.z*cp0.z;
    const GeoReal n1 = cp1.x*cp1.x+cp1.y*cp1.y+cp1.z*cp1.z;
    if (n0 < 1.0e-16f || n1 < 1.0e-16f) PRISM_PHYSICS_TRAP();
    GeoReal cosine = (cp0.x*cp1.x+cp0.y*cp1.y+cp0.z*cp1.z)/bp_native_sqrt(n0*n1);
    cosine = bp_native_max(-1.0f, bp_native_min(1.0f, cosine));
    GeoReal phi = bp_native_acos(cosine);
    if (v0.x*cp1.x+v0.y*cp1.y+v0.z*cp1.z < 0.0f) phi = -phi;
    return phi;
}

// Production occupied-to-occupied endpoint plane. This kernel is launched on
// the same stream immediately after the ordinary fused force node, whose
// corresponding LJ/bonded/exception parameters have been zeroed host-side.
// It is therefore an additive replacement, never a shadow/double-counted
// evaluator. All force, energy and derivative outputs remain device resident.
template<bool Compensated>
__device__ __forceinline__ void bp_native_lambda_endpoint_body(
    const float3* positions, float3* forces, int n_atoms,
    const int* dual_atom_indices, int n_dual_atoms,
    const LambdaDualAtomPlane* atom_planes,
    const LambdaDualBondPlane* bonds, int n_bonds,
    const LambdaDualAnglePlane* angles, int n_angles,
    const LambdaDualTorsionPlane* torsions, int n_torsions,
    const LambdaDualExceptionPlane* exceptions, int n_exceptions,
    const int* exclusion_list, const int* exclusion_offsets,
    const int* pairs14_list, const int* pairs14_offsets,
    const float* slot_values, const int* endpoint_occupancy_masks, int n_slots,
    float alpha_lj, float softcore_power, float coupling_power,
    float cutoff_sq, float box_x, float box_y, float box_z,
    double* pe_components, double* lambda_energy_atom, double* lambda_dudl_atom
,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault
) {
    using G=BpNativeGeometry<Compensated>;using GVec=typename G::Vec;using GReal=typename G::Real;
    if constexpr(Compensated) {
        if(!bp_native_coordinate_owner_valid(coordinate_owner,reinterpret_cast<const float*>(positions),
            n_atoms,BP_COORD_CURRENT_INITIALIZED,precision_fault))return;
    }

    const int task = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    const float3 box = make_float3(box_x, box_y, box_z);

    if (task < n_dual_atoms) {
        const int i = dual_atom_indices[task];
        if (i < 0 || i >= n_atoms) PRISM_PHYSICS_TRAP();
        const LambdaDualAtomPlane plane_i = atom_planes[i];
        if (plane_i.slot < 0 || plane_i.slot >= n_slots
            || (plane_i.present_a != 0 && plane_i.present_a != 1)
            || (plane_i.present_b != 0 && plane_i.present_b != 1)
            || (!plane_i.present_a && plane_i.epsilon_a != 0.0f)
            || (!plane_i.present_b && plane_i.epsilon_b != 0.0f)
            || plane_i.reserved != 0)
            PRISM_PHYSICS_TRAP();
        const float lambda = slot_values[plane_i.slot];
        const int slot_mask = endpoint_occupancy_masks[plane_i.slot];
        if (!isfinite(lambda) || lambda < 0.0f || lambda > 1.0f
            || slot_mask < 1 || slot_mask > 3) PRISM_PHYSICS_TRAP();
        for (int j = 0; j < n_atoms; ++j) {
            if (j == i) continue;
            const LambdaDualAtomPlane plane_j = atom_planes[j];
            if (plane_j.slot < -1 || plane_j.slot >= n_slots
                || (plane_j.present_a != 0 && plane_j.present_a != 1)
                || (plane_j.present_b != 0 && plane_j.present_b != 1)
                || (!plane_j.present_a && plane_j.epsilon_a != 0.0f)
                || (!plane_j.present_b && plane_j.epsilon_b != 0.0f)
                || plane_j.reserved != 0) PRISM_PHYSICS_TRAP();
            // All lambda-involving LJ is owned by this endpoint-aware node.
            // The ordinary fused node retains only non-lambda LJ. Pair-order
            // suppression below makes same- and cross-slot interactions occur
            // exactly once while the fragment-owned row uniquely owns
            // slot-environment interactions.
            if (plane_j.slot >= 0 && j < i) continue;
            if (lambda_csr_contains(i, j, exclusion_list, exclusion_offsets)
                || lambda_csr_contains(i, j, pairs14_list, pairs14_offsets)) continue;
            GVec displacement = G::make(
                G::load(positions,i,coordinate_owner,precision_fault).x-G::load(positions,j,coordinate_owner,precision_fault).x,
                G::load(positions,i,coordinate_owner,precision_fault).y-G::load(positions,j,coordinate_owner,precision_fault).y,
                G::load(positions,i,coordinate_owner,precision_fault).z-G::load(positions,j,coordinate_owner,precision_fault).z);
            displacement = minimum_image_orthorhombic_v1(displacement, box);
            int present_a = plane_i.present_a && plane_j.present_a;
            int present_b = plane_i.present_b && plane_j.present_b;
            float sigma_a = 0.5f*(plane_i.sigma_a+plane_j.sigma_a);
            float sigma_b = 0.5f*(plane_i.sigma_b+plane_j.sigma_b);
            float epsilon_a = sqrtf(plane_i.epsilon_a*plane_j.epsilon_a);
            float epsilon_b = sqrtf(plane_i.epsilon_b*plane_j.epsilon_b);
            // A persistent dummy slot retains its complete intramolecular
            // Hamiltonian at the EMPTY endpoint. Only slot-environment (and
            // explicitly declared cross-slot) interactions are decoupled.
            // Copying the occupied endpoint here makes same-slot LJ exactly
            // lambda-independent while preserving occupied-to-occupied
            // identity interpolation.
            const bool same_slot_internal =
                plane_j.slot >= 0 && plane_j.slot == plane_i.slot;
            if (same_slot_internal && slot_mask != 3 && present_a != present_b) {
                if (present_a) {
                    present_b = 1; sigma_b = sigma_a; epsilon_b = epsilon_a;
                } else {
                    present_a = 1; sigma_a = sigma_b; epsilon_a = epsilon_b;
                }
            }
            float3 fi = make_float3(0.0f, 0.0f, 0.0f);
            double energy = 0.0;
            double dudl = 0.0;
            double dudl_j = 0.0;
            if (plane_j.slot >= 0 && plane_j.slot != plane_i.slot) {
                const float lambda_j = slot_values[plane_j.slot];
                compute_lambda_dual_cross_slot_lj_v4_pair(
                    displacement, lambda,
                    plane_i.present_a, plane_i.sigma_a, plane_i.epsilon_a,
                    plane_i.present_b, plane_i.sigma_b, plane_i.epsilon_b,
                    lambda_j,
                    plane_j.present_a, plane_j.sigma_a, plane_j.epsilon_a,
                    plane_j.present_b, plane_j.sigma_b, plane_j.epsilon_b,
                    alpha_lj, softcore_power, coupling_power,
                    cutoff_sq, false, fi, energy, dudl, dudl_j);
            } else {
                if (present_a == 0 && present_b == 0) continue;
                compute_lambda_dual_endpoint_lj_v3_pair(
                    displacement, lambda,
                    present_a, sigma_a, epsilon_a,
                    present_b, sigma_b, epsilon_b,
                    alpha_lj, softcore_power, coupling_power,
                    cutoff_sq, false, fi, energy, dudl);
            }
            lambda_atomic_add_force(forces, i, fi);
            lambda_atomic_add_force(forces, j, make_float3(-fi.x, -fi.y, -fi.z));
            atomicAdd(pe_components+i, 0.5*energy);
            atomicAdd(pe_components+j, 0.5*energy);
            if (plane_j.slot >= 0 && plane_j.slot != plane_i.slot) {
                atomicAdd(lambda_energy_atom+i, 0.5*energy);
                atomicAdd(lambda_energy_atom+j, 0.5*energy);
                atomicAdd(lambda_dudl_atom+j, dudl_j);
            } else {
                atomicAdd(lambda_energy_atom+i, energy);
            }
            atomicAdd(lambda_dudl_atom+i, dudl);
        }
    }

    if (task < n_bonds) {
        const LambdaDualBondPlane term = bonds[task];
        if (term.i < 0 || term.i >= n_atoms || term.j < 0 || term.j >= n_atoms
            || term.i == term.j || term.slot < 0 || term.slot >= n_slots
            || (term.active_a != 0 && term.active_a != 1)
            || (term.active_b != 0 && term.active_b != 1)
            || term.reserved != 0
            || (!term.active_a && term.k_a != 0.0f)
            || (!term.active_b && term.k_b != 0.0f)) PRISM_PHYSICS_TRAP();
        const float lambda = slot_values[term.slot];
        const int slot_mask = endpoint_occupancy_masks[term.slot];
        if (!isfinite(lambda) || lambda < 0.0f || lambda > 1.0f
            || slot_mask < 1 || slot_mask > 3) PRISM_PHYSICS_TRAP();
        GVec pj = minimum_image_orthorhombic_v1(G::make(
            G::load(positions,term.j,coordinate_owner,precision_fault).x-G::load(positions,term.i,coordinate_owner,precision_fault).x,
            G::load(positions,term.j,coordinate_owner,precision_fault).y-G::load(positions,term.i,coordinate_owner,precision_fault).y,
            G::load(positions,term.j,coordinate_owner,precision_fault).z-G::load(positions,term.i,coordinate_owner,precision_fault).z), box);
        const GVec pi = G::make(0.0f, 0.0f, 0.0f);
        float3 fia=make_float3(0,0,0), fja=make_float3(0,0,0);
        float3 fib=make_float3(0,0,0), fjb=make_float3(0,0,0);
        const bool persistent_internal = slot_mask != 3;
        const float r0_a = (persistent_internal && !term.active_a && term.active_b) ? term.r0_b : term.r0_a;
        const float k_a = (persistent_internal && !term.active_a && term.active_b) ? term.k_b : term.k_a;
        const float r0_b = (persistent_internal && !term.active_b && term.active_a) ? term.r0_a : term.r0_b;
        const float k_b = (persistent_internal && !term.active_b && term.active_a) ? term.k_a : term.k_b;
        compute_bond_force(pi,pj,r0_a,k_a,fia,fja,nullptr,0,0);
        compute_bond_force(pi,pj,r0_b,k_b,fib,fjb,nullptr,0,0);
        const float3 fi=make_float3((1-lambda)*fia.x+lambda*fib.x,
            (1-lambda)*fia.y+lambda*fib.y,(1-lambda)*fia.z+lambda*fib.z);
        const float3 fj=make_float3((1-lambda)*fja.x+lambda*fjb.x,
            (1-lambda)*fja.y+lambda*fjb.y,(1-lambda)*fja.z+lambda*fjb.z);
        lambda_atomic_add_force(forces,term.i,fi); lambda_atomic_add_force(forces,term.j,fj);
        const double r=sqrt((double)pj.x*pj.x+(double)pj.y*pj.y+(double)pj.z*pj.z);
        const double ua=(double)k_a*(r-(double)r0_a)*(r-(double)r0_a);
        const double ub=(double)k_b*(r-(double)r0_b)*(r-(double)r0_b);
        const double u=(1.0-(double)lambda)*ua+(double)lambda*ub;
        atomicAdd(pe_components+term.i,0.5*u); atomicAdd(pe_components+term.j,0.5*u);
        atomicAdd(lambda_energy_atom+term.i,u); atomicAdd(lambda_dudl_atom+term.i,ub-ua);
    }

    if (task < n_angles) {
        const LambdaDualAnglePlane term=angles[task];
        if (term.i < 0 || term.i >= n_atoms || term.j < 0 || term.j >= n_atoms
            || term.k < 0 || term.k >= n_atoms || term.i == term.j || term.j == term.k
            || term.i == term.k || term.slot < 0 || term.slot >= n_slots
            || (term.active_a != 0 && term.active_a != 1)
            || (term.active_b != 0 && term.active_b != 1)
            || (!term.active_a && term.force_k_a != 0.0f)
            || (!term.active_b && term.force_k_b != 0.0f)) PRISM_PHYSICS_TRAP();
        const float lambda=slot_values[term.slot];
        const int slot_mask=endpoint_occupancy_masks[term.slot];
        if (!isfinite(lambda) || lambda < 0.0f || lambda > 1.0f
            || slot_mask < 1 || slot_mask > 3) PRISM_PHYSICS_TRAP();
        const GVec pj=G::make(0,0,0);
        const GVec pi=minimum_image_orthorhombic_v1(G::make(
            G::load(positions,term.i,coordinate_owner,precision_fault).x-G::load(positions,term.j,coordinate_owner,precision_fault).x,G::load(positions,term.i,coordinate_owner,precision_fault).y-G::load(positions,term.j,coordinate_owner,precision_fault).y,
            G::load(positions,term.i,coordinate_owner,precision_fault).z-G::load(positions,term.j,coordinate_owner,precision_fault).z),box);
        const GVec pk=minimum_image_orthorhombic_v1(G::make(
            G::load(positions,term.k,coordinate_owner,precision_fault).x-G::load(positions,term.j,coordinate_owner,precision_fault).x,G::load(positions,term.k,coordinate_owner,precision_fault).y-G::load(positions,term.j,coordinate_owner,precision_fault).y,
            G::load(positions,term.k,coordinate_owner,precision_fault).z-G::load(positions,term.j,coordinate_owner,precision_fault).z),box);
        float3 fia=make_float3(0,0,0),fja=make_float3(0,0,0),fka=make_float3(0,0,0);
        float3 fib=make_float3(0,0,0),fjb=make_float3(0,0,0),fkb=make_float3(0,0,0);
        const bool persistent_internal=slot_mask != 3;
        const float theta0_a = (persistent_internal && !term.active_a && term.active_b) ? term.theta0_b : term.theta0_a;
        const float force_k_a = (persistent_internal && !term.active_a && term.active_b) ? term.force_k_b : term.force_k_a;
        const float theta0_b = (persistent_internal && !term.active_b && term.active_a) ? term.theta0_a : term.theta0_b;
        const float force_k_b = (persistent_internal && !term.active_b && term.active_a) ? term.force_k_a : term.force_k_b;
        compute_angle_force(pi,pj,pk,theta0_a,force_k_a,fia,fja,fka,nullptr,0,0,0);
        compute_angle_force(pi,pj,pk,theta0_b,force_k_b,fib,fjb,fkb,nullptr,0,0,0);
        const float3 fi=make_float3((1-lambda)*fia.x+lambda*fib.x,(1-lambda)*fia.y+lambda*fib.y,(1-lambda)*fia.z+lambda*fib.z);
        const float3 fj=make_float3((1-lambda)*fja.x+lambda*fjb.x,(1-lambda)*fja.y+lambda*fjb.y,(1-lambda)*fja.z+lambda*fjb.z);
        const float3 fk=make_float3((1-lambda)*fka.x+lambda*fkb.x,(1-lambda)*fka.y+lambda*fkb.y,(1-lambda)*fka.z+lambda*fkb.z);
        lambda_atomic_add_force(forces,term.i,fi); lambda_atomic_add_force(forces,term.j,fj); lambda_atomic_add_force(forces,term.k,fk);
        const double ri=sqrt((double)pi.x*pi.x+(double)pi.y*pi.y+(double)pi.z*pi.z);
        const double rk=sqrt((double)pk.x*pk.x+(double)pk.y*pk.y+(double)pk.z*pk.z);
        double cosine=((double)pi.x*pk.x+(double)pi.y*pk.y+(double)pi.z*pk.z)/(ri*rk);
        cosine=fmax(-1.0,fmin(1.0,cosine)); const double theta=acos(cosine);
        const double ua=(double)force_k_a*(theta-(double)theta0_a)*(theta-(double)theta0_a);
        const double ub=(double)force_k_b*(theta-(double)theta0_b)*(theta-(double)theta0_b);
        const double u=(1.0-(double)lambda)*ua+(double)lambda*ub;
        atomicAdd(pe_components+term.i,u/3.0); atomicAdd(pe_components+term.j,u/3.0); atomicAdd(pe_components+term.k,u/3.0);
        atomicAdd(lambda_energy_atom+term.i,u); atomicAdd(lambda_dudl_atom+term.i,ub-ua);
    }

    if (task < n_torsions) {
        const LambdaDualTorsionPlane term=torsions[task];
        if (term.i < 0 || term.i >= n_atoms || term.j < 0 || term.j >= n_atoms
            || term.k < 0 || term.k >= n_atoms || term.l < 0 || term.l >= n_atoms
            || term.slot < 0 || term.slot >= n_slots || term.periodicity <= 0
            || (term.active_a != 0 && term.active_a != 1)
            || (term.active_b != 0 && term.active_b != 1)
            || (term.improper != 0 && term.improper != 1)
            || term.reserved != 0
            || (!term.active_a && term.force_k_a != 0.0f)
            || (!term.active_b && term.force_k_b != 0.0f)) PRISM_PHYSICS_TRAP();
        const float lambda=slot_values[term.slot];
        const int slot_mask=endpoint_occupancy_masks[term.slot];
        if (!isfinite(lambda) || lambda < 0.0f || lambda > 1.0f
            || slot_mask < 1 || slot_mask > 3) PRISM_PHYSICS_TRAP();
        const GVec pi=G::make(0,0,0);
        const GVec pj=minimum_image_orthorhombic_v1(G::make(
            G::load(positions,term.j,coordinate_owner,precision_fault).x-G::load(positions,term.i,coordinate_owner,precision_fault).x,G::load(positions,term.j,coordinate_owner,precision_fault).y-G::load(positions,term.i,coordinate_owner,precision_fault).y,
            G::load(positions,term.j,coordinate_owner,precision_fault).z-G::load(positions,term.i,coordinate_owner,precision_fault).z),box);
        const GVec dkj=minimum_image_orthorhombic_v1(G::make(
            G::load(positions,term.k,coordinate_owner,precision_fault).x-G::load(positions,term.j,coordinate_owner,precision_fault).x,G::load(positions,term.k,coordinate_owner,precision_fault).y-G::load(positions,term.j,coordinate_owner,precision_fault).y,
            G::load(positions,term.k,coordinate_owner,precision_fault).z-G::load(positions,term.j,coordinate_owner,precision_fault).z),box);
        const GVec pk=G::make(pj.x+dkj.x,pj.y+dkj.y,pj.z+dkj.z);
        const GVec dlk=minimum_image_orthorhombic_v1(G::make(
            G::load(positions,term.l,coordinate_owner,precision_fault).x-G::load(positions,term.k,coordinate_owner,precision_fault).x,G::load(positions,term.l,coordinate_owner,precision_fault).y-G::load(positions,term.k,coordinate_owner,precision_fault).y,
            G::load(positions,term.l,coordinate_owner,precision_fault).z-G::load(positions,term.k,coordinate_owner,precision_fault).z),box);
        const GVec pl=G::make(pk.x+dlk.x,pk.y+dlk.y,pk.z+dlk.z);
        float3 fia=make_float3(0,0,0),fja=make_float3(0,0,0),fka=make_float3(0,0,0),fla=make_float3(0,0,0);
        float3 fib=make_float3(0,0,0),fjb=make_float3(0,0,0),fkb=make_float3(0,0,0),flb=make_float3(0,0,0);
        const bool persistent_internal=slot_mask != 3;
        const float phase_a = (persistent_internal && !term.active_a && term.active_b) ? term.phase_b : term.phase_a;
        const float force_k_a = (persistent_internal && !term.active_a && term.active_b) ? term.force_k_b : term.force_k_a;
        const float phase_b = (persistent_internal && !term.active_b && term.active_a) ? term.phase_a : term.phase_b;
        const float force_k_b = (persistent_internal && !term.active_b && term.active_a) ? term.force_k_a : term.force_k_b;
        compute_dihedral_force(pi,pj,pk,pl,term.periodicity,phase_a,force_k_a,fia,fja,fka,fla,nullptr,0,0,0,0);
        compute_dihedral_force(pi,pj,pk,pl,term.periodicity,phase_b,force_k_b,fib,fjb,fkb,flb,nullptr,0,0,0,0);
        const float3 fi=make_float3((1-lambda)*fia.x+lambda*fib.x,(1-lambda)*fia.y+lambda*fib.y,(1-lambda)*fia.z+lambda*fib.z);
        const float3 fj=make_float3((1-lambda)*fja.x+lambda*fjb.x,(1-lambda)*fja.y+lambda*fjb.y,(1-lambda)*fja.z+lambda*fjb.z);
        const float3 fk=make_float3((1-lambda)*fka.x+lambda*fkb.x,(1-lambda)*fka.y+lambda*fkb.y,(1-lambda)*fka.z+lambda*fkb.z);
        const float3 fl=make_float3((1-lambda)*fla.x+lambda*flb.x,(1-lambda)*fla.y+lambda*flb.y,(1-lambda)*fla.z+lambda*flb.z);
        lambda_atomic_add_force(forces,term.i,fi); lambda_atomic_add_force(forces,term.j,fj); lambda_atomic_add_force(forces,term.k,fk); lambda_atomic_add_force(forces,term.l,fl);
        const double phi=(double)lambda_torsion_phi_v1(pi,pj,pk,pl);
        const double ua=(double)force_k_a*(1.0+cos((double)term.periodicity*phi-(double)phase_a));
        const double ub=(double)force_k_b*(1.0+cos((double)term.periodicity*phi-(double)phase_b));
        const double u=(1.0-(double)lambda)*ua+(double)lambda*ub;
        atomicAdd(pe_components+term.i,0.25*u); atomicAdd(pe_components+term.j,0.25*u); atomicAdd(pe_components+term.k,0.25*u); atomicAdd(pe_components+term.l,0.25*u);
        atomicAdd(lambda_energy_atom+term.i,u); atomicAdd(lambda_dudl_atom+term.i,ub-ua);
    }

    if (task < n_exceptions) {
        const LambdaDualExceptionPlane term=exceptions[task];
        if (term.i < 0 || term.i >= n_atoms || term.j < 0 || term.j >= n_atoms
            || term.i == term.j || term.slot < 0 || term.slot >= n_slots
            || (term.present_a != 0 && term.present_a != 1)
            || (term.present_b != 0 && term.present_b != 1)
            || term.reserved != 0
            || (!term.present_a && (term.charge_product_a != 0.0f || term.epsilon_a != 0.0f))
            || (!term.present_b && (term.charge_product_b != 0.0f || term.epsilon_b != 0.0f))
            || (!term.present_a && !term.present_b)
            || !lambda_csr_contains(term.i, term.j, pairs14_list, pairs14_offsets))
            PRISM_PHYSICS_TRAP();
        const float lambda=slot_values[term.slot];
        const int slot_mask=endpoint_occupancy_masks[term.slot];
        if (!isfinite(lambda) || lambda < 0.0f || lambda > 1.0f
            || slot_mask < 1 || slot_mask > 3) PRISM_PHYSICS_TRAP();
        GVec displacement=minimum_image_orthorhombic_v1(G::make(
            G::load(positions,term.i,coordinate_owner,precision_fault).x-G::load(positions,term.j,coordinate_owner,precision_fault).x,G::load(positions,term.i,coordinate_owner,precision_fault).y-G::load(positions,term.j,coordinate_owner,precision_fault).y,
            G::load(positions,term.i,coordinate_owner,precision_fault).z-G::load(positions,term.j,coordinate_owner,precision_fault).z),box);
        float3 fi=make_float3(0,0,0); double lj=0.0,dudl_lj=0.0;
        int present_a = term.present_a;
        int present_b = term.present_b;
        float sigma_a = term.sigma_a;
        float sigma_b = term.sigma_b;
        float epsilon_a = term.epsilon_a;
        float epsilon_b = term.epsilon_b;
        double charge_product_a = (double)term.charge_product_a;
        double charge_product_b = (double)term.charge_product_b;
        if (slot_mask != 3 && present_a != present_b) {
            if (present_a) {
                present_b = 1; sigma_b = sigma_a; epsilon_b = epsilon_a;
                charge_product_b = charge_product_a;
            } else {
                present_a = 1; sigma_a = sigma_b; epsilon_a = epsilon_b;
                charge_product_a = charge_product_b;
            }
        }
        compute_lambda_dual_endpoint_lj_v3_pair(displacement,lambda,
            present_a,sigma_a,epsilon_a,present_b,sigma_b,epsilon_b,
            alpha_lj,softcore_power,coupling_power,cutoff_sq,true,fi,lj,dudl_lj);
        const double r=sqrt((double)displacement.x*displacement.x+(double)displacement.y*displacement.y+(double)displacement.z*displacement.z);
        if (r < 1.0e-8) PRISM_PHYSICS_TRAP();
        const double q=(1.0-(double)lambda)*charge_product_a+(double)lambda*charge_product_b;
        const double dq=charge_product_b-charge_product_a;
        const double uq=(double)COULOMB_CONSTANT*q/r;
        const double fq=(double)COULOMB_CONSTANT*q/(r*r*r);
        fi.x+=(float)(fq*(double)displacement.x); fi.y+=(float)(fq*(double)displacement.y); fi.z+=(float)(fq*(double)displacement.z);
        const double energy=lj+uq; const double dudl=dudl_lj+(double)COULOMB_CONSTANT*dq/r;
        lambda_atomic_add_force(forces,term.i,fi); lambda_atomic_add_force(forces,term.j,make_float3(-fi.x,-fi.y,-fi.z));
        atomicAdd(pe_components+term.i,0.5*energy); atomicAdd(pe_components+term.j,0.5*energy);
        atomicAdd(lambda_energy_atom+term.i,energy); atomicAdd(lambda_dudl_atom+term.i,dudl);
    }
}

extern "C" __global__ void nhs_lambda_dual_endpoint_hamiltonian_v1(
    const float3* positions, float3* forces, int n_atoms,
    const int* dual_atom_indices, int n_dual_atoms,
    const LambdaDualAtomPlane* atom_planes,
    const LambdaDualBondPlane* bonds, int n_bonds,
    const LambdaDualAnglePlane* angles, int n_angles,
    const LambdaDualTorsionPlane* torsions, int n_torsions,
    const LambdaDualExceptionPlane* exceptions, int n_exceptions,
    const int* exclusion_list, const int* exclusion_offsets,
    const int* pairs14_list, const int* pairs14_offsets,
    const float* slot_values, const int* endpoint_occupancy_masks, int n_slots,
    float alpha_lj, float softcore_power, float coupling_power,
    float cutoff_sq, float box_x, float box_y, float box_z,
    double* pe_components, double* lambda_energy_atom, double* lambda_dudl_atom
) {
    bp_native_lambda_endpoint_body<false>(positions, forces, n_atoms, dual_atom_indices, n_dual_atoms, atom_planes, bonds, n_bonds, angles, n_angles, torsions, n_torsions, exceptions, n_exceptions, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, slot_values, endpoint_occupancy_masks, n_slots, alpha_lj, softcore_power, coupling_power, cutoff_sq, box_x, box_y, box_z, pe_components, lambda_energy_atom, lambda_dudl_atom, nullptr, nullptr);
}

extern "C" __global__ void nhs_lambda_dual_endpoint_hamiltonian_compensated_v1(
    const float3* positions, float3* forces, int n_atoms,
    const int* dual_atom_indices, int n_dual_atoms,
    const LambdaDualAtomPlane* atom_planes,
    const LambdaDualBondPlane* bonds, int n_bonds,
    const LambdaDualAnglePlane* angles, int n_angles,
    const LambdaDualTorsionPlane* torsions, int n_torsions,
    const LambdaDualExceptionPlane* exceptions, int n_exceptions,
    const int* exclusion_list, const int* exclusion_offsets,
    const int* pairs14_list, const int* pairs14_offsets,
    const float* slot_values, const int* endpoint_occupancy_masks, int n_slots,
    float alpha_lj, float softcore_power, float coupling_power,
    float cutoff_sq, float box_x, float box_y, float box_z,
    double* pe_components, double* lambda_energy_atom, double* lambda_dudl_atom
,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault
) {
    bp_native_lambda_endpoint_body<true>(positions, forces, n_atoms, dual_atom_indices, n_dual_atoms, atom_planes, bonds, n_bonds, angles, n_angles, torsions, n_torsions, exceptions, n_exceptions, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, slot_values, endpoint_occupancy_masks, n_slots, alpha_lj, softcore_power, coupling_power, cutoff_sq, box_x, box_y, box_z, pe_components, lambda_energy_atom, lambda_dudl_atom, coordinate_owner, precision_fault);
}

// Direct validation entrypoint for finite-difference and endpoint-equivalence
// acceptance tests without running a trajectory.
extern "C" __global__ void nhs_lambda_dual_endpoint_lj_reference_v3(
    float dx,float dy,float dz,float lambda,
    int present_a,float sigma_a,float epsilon_a,
    int present_b,float sigma_b,float epsilon_b,
    float alpha_lj,float softcore_power,float coupling_power,float cutoff,
    double* potential_out,float* force_out,double* dudl_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float3 force=make_float3(0,0,0); double energy=0.0,dudl=0.0;
    compute_lambda_dual_endpoint_lj_v3_pair(make_float3(dx,dy,dz),lambda,
        present_a,sigma_a,epsilon_a,present_b,sigma_b,epsilon_b,
        alpha_lj,softcore_power,coupling_power,cutoff*cutoff,false,force,energy,dudl);
    potential_out[0]=energy; force_out[0]=force.x; force_out[1]=force.y; force_out[2]=force.z; dudl_out[0]=dudl;
}

extern "C" __global__ void nhs_lambda_dual_cross_slot_lj_reference_v4(
    float dx,float dy,float dz,float lambda_i,
    int present_ia,float sigma_ia,float epsilon_ia,
    int present_ib,float sigma_ib,float epsilon_ib,
    float lambda_j,
    int present_ja,float sigma_ja,float epsilon_ja,
    int present_jb,float sigma_jb,float epsilon_jb,
    float alpha_lj,float softcore_power,float coupling_power,float cutoff,
    double* potential_out,float* force_out,double* dudl_i_out,double* dudl_j_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float3 force=make_float3(0,0,0); double energy=0.0,dudl_i=0.0,dudl_j=0.0;
    compute_lambda_dual_cross_slot_lj_v4_pair(
        make_float3(dx,dy,dz),lambda_i,
        present_ia,sigma_ia,epsilon_ia,present_ib,sigma_ib,epsilon_ib,
        lambda_j,
        present_ja,sigma_ja,epsilon_ja,present_jb,sigma_jb,epsilon_jb,
        alpha_lj,softcore_power,coupling_power,cutoff*cutoff,false,
        force,energy,dudl_i,dudl_j);
    potential_out[0]=energy; force_out[0]=force.x; force_out[1]=force.y;
    force_out[2]=force.z; dudl_i_out[0]=dudl_i; dudl_j_out[0]=dudl_j;
}

template<class GeoVec>
__device__ __forceinline__ void compute_charge_product_ewald_realspace_v2(
    GeoVec displacement, double charge_product, double d_charge_product,
    float beta, float3& force_i, double& potential, double& du_dlambda);
template<class GeoVec>
__device__ __forceinline__ void compute_dynamic_charge_ewald_exception_v2(
    GeoVec displacement,
    float regular_q_product, float d_regular_q_product,
    float exception_q_product, float d_exception_q_product,
    float beta, float3& force_i, double& potential, double& du_dlambda);

extern "C" __global__ void nhs_lambda_same_slot_ewald_reference_v3(
    float dx,float dy,float dz,float lambda,int endpoint_mask,
    float qa_i,float qa_j,float qb_i,float qb_j,
    float exact_qa,float exact_qb,float beta,int exact_pair,
    double* potential_out,float* force_out,double* dudl_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const double regular_a=(double)qa_i*(double)qa_j;
    const double regular_b=(double)qb_i*(double)qb_j;
    double regular=0.0,dregular=0.0;
    if (endpoint_mask == 1) regular=regular_a;
    else if (endpoint_mask == 2) regular=regular_b;
    else if (endpoint_mask == 3) {
        regular=(1.0-(double)lambda)*regular_a+(double)lambda*regular_b;
        dregular=regular_b-regular_a;
    } else PRISM_PHYSICS_TRAP();
    double exact=0.0,dexact=0.0;
    if (endpoint_mask == 1) exact=(double)exact_qa;
    else if (endpoint_mask == 2) exact=(double)exact_qb;
    else {
        exact=(1.0-(double)lambda)*(double)exact_qa+(double)lambda*(double)exact_qb;
        dexact=(double)exact_qb-(double)exact_qa;
    }
    float3 force=make_float3(0,0,0); double energy=0.0,dudl=0.0;
    if (exact_pair != 0) {
        compute_dynamic_charge_ewald_exception_v2(
            make_float3(dx,dy,dz),(float)regular,(float)dregular,
            (float)exact,(float)dexact,beta,force,energy,dudl);
    } else {
        compute_charge_product_ewald_realspace_v2(
            make_float3(dx,dy,dz),regular,dregular,beta,force,energy,dudl);
    }
    potential_out[0]=energy; force_out[0]=force.x; force_out[1]=force.y;
    force_out[2]=force.z; dudl_out[0]=dudl;
}

extern "C" __global__ void nhs_reverse_velocities(float3* velocities, int n_atoms) {
    const int atom=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if (atom < n_atoms) {
        velocities[atom].x=-velocities[atom].x;
        velocities[atom].y=-velocities[atom].y;
        velocities[atom].z=-velocities[atom].z;
    }
}

// q_i(lambda) q_j(lambda) erfc(beta*r)/r. Current charges and their exact
// derivatives are inputs, so energy, force and dU/dlambda share one expression.
template<class GeoVec>
__device__ __forceinline__ void compute_dynamic_charge_ewald_realspace_v2(
    GeoVec displacement, float qi, float dqi, float qj, float dqj, float beta,
    float3& force_i, double& potential, double& du_dlambda
) {
    const double r2 = (double)displacement.x*displacement.x
                    + (double)displacement.y*displacement.y
                    + (double)displacement.z*displacement.z;
    if (r2 < 1.0e-16) return;
    const double r = sqrt(r2);
    const double inv_r = 1.0 / r;
    const double br = (double)beta * r;
    const double erfc_br = erfc(br);
    const double q = (double)qi * (double)qj;
    const double dq = (double)dqi * (double)qj + (double)qi * (double)dqj;
    const double radial = erfc_br * inv_r;
    const double prefactor = (double)COULOMB_CONSTANT;
    const double d_radial_dr = -2.0 * (double)beta / sqrt((double)M_PI)
        * exp(-br*br) * inv_r - erfc_br * inv_r * inv_r;
    const double radial_force = -prefactor * q * d_radial_dr;
    force_i.x += (float)(radial_force * (double)displacement.x * inv_r);
    force_i.y += (float)(radial_force * (double)displacement.y * inv_r);
    force_i.z += (float)(radial_force * (double)displacement.z * inv_r);
    potential += prefactor * q * radial;
    du_dlambda += prefactor * dq * radial;
}

// Same Ewald real-space operator expressed directly in charge-product space.
// This is required for occupied-to-occupied endpoint interpolation because
// (1-lambda) qA_i qA_j + lambda qB_i qB_j is not generally factorable as one
// pair of linearly interpolated atomic charges.
template<class GeoVec>
__device__ __forceinline__ void compute_charge_product_ewald_realspace_v2(
    GeoVec displacement, double charge_product, double d_charge_product,
    float beta, float3& force_i, double& potential, double& du_dlambda
) {
    const double r2 = (double)displacement.x*displacement.x
                    + (double)displacement.y*displacement.y
                    + (double)displacement.z*displacement.z;
    if (r2 < 1.0e-16) return;
    const double r = sqrt(r2);
    const double inv_r = 1.0 / r;
    const double br = (double)beta * r;
    const double erfc_br = erfc(br);
    const double radial = erfc_br * inv_r;
    const double prefactor = (double)COULOMB_CONSTANT;
    const double d_radial_dr = -2.0 * (double)beta / sqrt((double)M_PI)
        * exp(-br*br) * inv_r - erfc_br * inv_r * inv_r;
    const double radial_force = -prefactor * charge_product * d_radial_dr;
    force_i.x += (float)(radial_force * (double)displacement.x * inv_r);
    force_i.y += (float)(radial_force * (double)displacement.y * inv_r);
    force_i.z += (float)(radial_force * (double)displacement.z * inv_r);
    potential += prefactor * charge_product * radial;
    du_dlambda += prefactor * d_charge_product * radial;
}

__device__ __forceinline__ void lambda_desired_internal_charge_product_v1(
    int atom_i, int atom_j, int slot, float lambda,
    const float* endpoint_a_charges, const float* endpoint_b_charges,
    const int* endpoint_occupancy_masks,
    double& charge_product, double& d_charge_product
) {
    const int mask = endpoint_occupancy_masks[slot];
    if (mask < 1 || mask > 3 || !isfinite(lambda) || lambda < 0.0f || lambda > 1.0f)
        PRISM_PHYSICS_TRAP();
    const double qa = (double)endpoint_a_charges[atom_i]
        * (double)endpoint_a_charges[atom_j];
    const double qb = (double)endpoint_b_charges[atom_i]
        * (double)endpoint_b_charges[atom_j];
    if (mask == 1) {
        charge_product = qa;
        d_charge_product = 0.0;
    } else if (mask == 2) {
        charge_product = qb;
        d_charge_product = 0.0;
    } else {
        charge_product = (1.0-(double)lambda)*qa + (double)lambda*qb;
        d_charge_product = qb-qa;
    }
}

// Reciprocal exclusion/scaled-1-4 correction for an ordinary pair omitted
// from the erfc neighbor loop: q_exception/r - q_regular*erf(beta*r)/r.
template<class GeoVec>
__device__ __forceinline__ void compute_dynamic_charge_ewald_exception_v2(
    GeoVec displacement,
    float regular_q_product, float d_regular_q_product,
    float exception_q_product, float d_exception_q_product,
    float beta, float3& force_i, double& potential, double& du_dlambda
) {
    const double r2 = (double)displacement.x*displacement.x
                    + (double)displacement.y*displacement.y
                    + (double)displacement.z*displacement.z;
    if (r2 < 1.0e-16) return;
    const double r = sqrt(r2);
    const double inv_r = 1.0 / r;
    const double br = (double)beta * r;
    const double erf_br = erf(br);
    const double numerator = (double)exception_q_product
        - (double)regular_q_product * erf_br;
    const double d_numerator = (double)d_exception_q_product
        - (double)d_regular_q_product * erf_br;
    const double d_erf_dr = 2.0 * (double)beta / sqrt((double)M_PI) * exp(-br*br);
    const double prefactor = (double)COULOMB_CONSTANT;
    const double dpotential_dr = prefactor
        * (-(double)regular_q_product*d_erf_dr*inv_r - numerator*inv_r*inv_r);
    const double radial_force = -dpotential_dr;
    force_i.x += (float)(radial_force * (double)displacement.x * inv_r);
    force_i.y += (float)(radial_force * (double)displacement.y * inv_r);
    force_i.z += (float)(radial_force * (double)displacement.z * inv_r);
    potential += prefactor * numerator * inv_r;
    du_dlambda += prefactor * d_numerator * inv_r;
}

// Direct V2 device acceptance entrypoint. mode=0 evaluates ordinary erfc;
// mode=1 evaluates the exact reciprocal exclusion/scaled-pair correction.
extern "C" __global__ void nhs_lambda_fragment_ewald_pair_reference_v2(
    float dx, float dy, float dz,
    float box_x, float box_y, float box_z,
    float qi, float dqi, float qj, float dqj,
    float exception_q_product, float d_exception_q_product,
    float beta, int mode,
    double* potential_out, float* force_out, double* dudl_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float3 displacement = minimum_image_orthorhombic_v1(
        make_float3(dx, dy, dz), make_float3(box_x, box_y, box_z));
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    double potential = 0.0;
    double dudl = 0.0;
    if (mode == 0) {
        compute_dynamic_charge_ewald_realspace_v2(
            displacement, qi, dqi, qj, dqj, beta, force, potential, dudl);
    } else if (mode == 1) {
        const float regular = qi*qj;
        const float dregular = dqi*qj + qi*dqj;
        compute_dynamic_charge_ewald_exception_v2(
            displacement, regular, dregular,
            exception_q_product, d_exception_q_product,
            beta, force, potential, dudl);
    } else {
        PRISM_PHYSICS_TRAP();
    }
    potential_out[0] = potential;
    force_out[0] = force.x;
    force_out[1] = force.y;
    force_out[2] = force.z;
    dudl_out[0] = dudl;
}

// Direct validation entrypoint for the production device function above.
// Acceptance tests invoke this symbol and compare U/F/dUdl with the f64 CPU
// reference and independent central differences.
extern "C" __global__ void nhs_lambda_fragment_pair_reference(
    float dx, float dy, float dz,
    float lambda_pair, float charge_product, float sigma, float epsilon,
    float alpha_lj, float alpha_q_a2, float softcore_power, float coupling_power,
    float cutoff,
    double* potential_out, float* force_out, double* dudl_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float3 pi = make_float3(dx, dy, dz);
    const float3 pj = make_float3(0.0f, 0.0f, 0.0f);
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    double potential = 0.0;
    double dudl = 0.0;
    compute_lambda_softcore_pair(
        pi, pj, charge_product, sigma, epsilon, lambda_pair,
        alpha_lj, alpha_q_a2, softcore_power, coupling_power,
        cutoff * cutoff, false, force, potential, dudl);
    potential_out[0] = potential;
    force_out[0] = force.x;
    force_out[1] = force.y;
    force_out[2] = force.z;
    dudl_out[0] = dudl;
}

// Controlled NCMC-boundary reduction.  One block owns one stable runtime
// slot; lane zero performs a fixed atom-index-order sum, making the byte-level
// result replayable across launches rather than atomic-order dependent.
extern "C" __global__ void nhs_lambda_reduce_observables(
    const int* atom_slot,
    const double* atom_energy,
    const double* atom_dudl,
    int n_atoms,
    int n_slots,
    double* slot_energy,
    double* slot_dudl,
    const float3* velocities,
    const float* masses,
    const double* total_pe_atom,
    double* total_potential,
    double* total_kinetic
) {
    const int slot = (int)blockIdx.x;
    if (slot >= n_slots || threadIdx.x != 0) return;
    double e = 0.0;
    double d = 0.0;
    for (int atom = 0; atom < n_atoms; ++atom) {
        if (atom_slot[atom] == slot) {
            e += atom_energy[atom];
            d += atom_dudl[atom];
        }
    }
    slot_energy[slot] = e;
    slot_dudl[slot] = d;
    if (slot == 0) {
        double pe = 0.0;
        double ke = 0.0;
        for (int atom = 0; atom < n_atoms; ++atom) {
            const float3 v = velocities[atom];
            pe += total_pe_atom[atom];
            ke += 0.5 * (double)masses[atom]
                * ((double)v.x * v.x + (double)v.y * v.y + (double)v.z * v.z)
                / (double)AMBER_KCAL_TO_AMU_A2_PS2;
        }
        total_potential[0] = pe;
        total_kinetic[0] = ke;
    }
}

// ============================================================================
// LANGEVIN THERMOSTAT WITH DYNAMIC TEMPERATURE
// ============================================================================

__device__ void langevin_thermostat(
    float3& velocity,
    float mass,
    float target_temp,
    float gamma,
    float dt,
    curandState* rng_state,
    double* nonham_work_component
) {
    const float3 velocity_before = velocity;
    // Friction coefficient
    float c1 = expf(-gamma * dt);
    // AMBER stores energy in kcal/mol, mass in amu, and velocity in A/ps.
    // Convert kBT to amu*A^2/ps^2 before evaluating fluctuation-dissipation.
    float c2 = sqrtf((1.0f - c1*c1) * AMBER_KCAL_TO_AMU_A2_PS2
                     * KB_KCAL_MOL_K * target_temp / mass);

    // Apply Langevin dynamics
    velocity.x = c1 * velocity.x + c2 * curand_normal(rng_state);
    velocity.y = c1 * velocity.y + c2 * curand_normal(rng_state);
    velocity.z = c1 * velocity.z + c2 * curand_normal(rng_state);
    if (nonham_work_component != nullptr) {
        const double before_sq = (double)velocity_before.x * (double)velocity_before.x
                               + (double)velocity_before.y * (double)velocity_before.y
                               + (double)velocity_before.z * (double)velocity_before.z;
        const double after_sq = (double)velocity.x * (double)velocity.x
                              + (double)velocity.y * (double)velocity.y
                              + (double)velocity.z * (double)velocity.z;
        *nonham_work_component += 0.5 * (double)mass * (after_sq - before_sq)
                                / (double)AMBER_KCAL_TO_AMU_A2_PS2;
    }
}

// ============================================================================
// AMBER-UNIT FORCE CLAMP AND SHAKE/RATTLE CONSTRAINT HELPERS
// ============================================================================

__device__ __forceinline__ void apply_force_half_kick_with_intervention_work(
    float3& velocity,
    const float3 physical_force,
    float force_scale,
    float mass,
    float dt,
    double* nonham_work_component
) {
    const float raw_force[3]={physical_force.x,physical_force.y,physical_force.z};
    float used[3];
    prism_asc_used_force(raw_force,force_scale,MAX_INTEGRATOR_FORCE_KCAL_MOL_A,used);
    const float3 used_force=make_float3(used[0],used[1],used[2]);
    const float magnitude = sqrtf((physical_force.x*force_scale)*(physical_force.x*force_scale)
        +(physical_force.y*force_scale)*(physical_force.y*force_scale)
        +(physical_force.z*force_scale)*(physical_force.z*force_scale));
    const float clamp_scale = magnitude > MAX_INTEGRATOR_FORCE_KCAL_MOL_A
        ? MAX_INTEGRATOR_FORCE_KCAL_MOL_A / magnitude : 1.0f;
    const float kick = 0.5f * dt * AMBER_KCAL_TO_AMU_A2_PS2 / mass;
    const float3 raw_candidate = make_float3(
        velocity.x + kick * physical_force.x,
        velocity.y + kick * physical_force.y,
        velocity.z + kick * physical_force.z);
    const float3 used_candidate = make_float3(
        velocity.x + kick * used_force.x,
        velocity.y + kick * used_force.y,
        velocity.z + kick * used_force.z);
    if (nonham_work_component != nullptr && (force_scale != 1.0f || clamp_scale != 1.0f)) {
        const double raw_sq = (double)raw_candidate.x * (double)raw_candidate.x
                            + (double)raw_candidate.y * (double)raw_candidate.y
                            + (double)raw_candidate.z * (double)raw_candidate.z;
        const double used_sq = (double)used_candidate.x * (double)used_candidate.x
                             + (double)used_candidate.y * (double)used_candidate.y
                             + (double)used_candidate.z * (double)used_candidate.z;
        *nonham_work_component += 0.5 * (double)mass * (used_sq - raw_sq)
                                / (double)AMBER_KCAL_TO_AMU_A2_PS2;
    }
    velocity = used_candidate;
}

// Exact Hamiltonian half-kick used exclusively by the admitted gamma-zero
// NCMC graph. It contains no REST2 scaling, force clamp or work correction.
// The reversible guard separately requires the physical atom-lambda plane.
__device__ __forceinline__ void apply_exact_force_half_kick(
    float3& velocity,
    const float3 physical_force,
    float mass,
    float dt
) {
    const float kick = 0.5f * dt * AMBER_KCAL_TO_AMU_A2_PS2 / mass;
    velocity.x += kick * physical_force.x;
    velocity.y += kick * physical_force.y;
    velocity.z += kick * physical_force.z;
}


// Explicit original-May numerical operator. Its unconventional no-418.4
// propagation is an identified historical algorithm, not an SI/unit-validation
// claim. Only the hash-bound signed historical model can select these nodes.
__device__ bool prism_may_original_first_kick_bath_drift_v1(
    float3& x,float3& x_start,float3& v,const float3 f,float mass,
    float lambda,float temp,float gamma,float dt,curandState* rng,
    double* pe,double* work) {
    if(!(isfinite(mass)&&mass>0.0f&&isfinite(temp)&&temp>0.0f&&
         isfinite(gamma)&&gamma>=0.0f&&isfinite(dt)&&dt>0.0f&&
         isfinite(lambda)))return false;
    float3 used=make_float3(f.x*lambda,f.y*lambda,f.z*lambda);
    const float magnitude=sqrtf(used.x*used.x+used.y*used.y+used.z*used.z);
    if(!isfinite(magnitude))return false;
    if(pe)*pe*=double(lambda);
    if(magnitude>1000.0f) {
        const float scale=1000.0f/magnitude;
        const float3 clamped=make_float3(used.x*scale,used.y*scale,used.z*scale);
        if(work)*work-=(((double)(used.x-clamped.x)*(double)v.x+
            (double)(used.y-clamped.y)*(double)v.y+
            (double)(used.z-clamped.z)*(double)v.z)*(double)dt)/418.4;
        used=clamped;
    }
    const float inv_mass=1.0f/mass;
    v.x+=0.5f*dt*used.x*inv_mass;
    v.y+=0.5f*dt*used.y*inv_mass;
    v.z+=0.5f*dt*used.z*inv_mass;
    const float c1=expf(-gamma*dt);
    const float c2=sqrtf((1.0f-c1*c1)*0.001987204f*temp/mass);
    // Exactly one full bath, xyz draw order; even gamma zero consumes the
    // original draws. The strict reversible NCMC path cannot select this mode.
    v.x=c1*v.x+c2*curand_normal(rng);
    v.y=c1*v.y+c2*curand_normal(rng);
    v.z=c1*v.z+c2*curand_normal(rng);
    const float thermal=sqrtf(0.001987204f*temp/fmaxf(mass,1.0f));
    const float maximum=fmaxf(thermal*20.0f,2.0f);
    const float speed=sqrtf(v.x*v.x+v.y*v.y+v.z*v.z);
    if(!isfinite(speed))return false;
    if(speed>maximum) {
        const float ratio=maximum/speed;const float3 old=v;
        v=make_float3(old.x*ratio,old.y*ratio,old.z*ratio);
        if(work)*work+=0.5*(double)mass*((double)v.x*v.x+(double)v.y*v.y+
            (double)v.z*v.z-(double)old.x*old.x-(double)old.y*old.y-(double)old.z*old.z)/418.4;
    }
    x_start=x;
    x.x+=dt*v.x;x.y+=dt*v.y;x.z+=dt*v.z;
    return isfinite(x.x)&&isfinite(x.y)&&isfinite(x.z)&&(!pe||isfinite(*pe))&&(!work||isfinite(*work));
}

// The original selected radial, position-only projection. It does not solve
// the different old-bond-direction quadratic and does not alter velocities.
// The original ten-iteration/absolute-residual acceptance is retained; only
// nonfinite, degenerate or invalid inputs trap, never a fallback from SHAKE.
__device__ bool prism_may_original_position_projection_v1(
    float3* x,const HCluster cluster) {
    if(cluster.central_atom<0||cluster.n_hydrogens<0||cluster.n_hydrogens>3||
       !isfinite(cluster.inv_mass_central)||!isfinite(cluster.inv_mass_h)||
       cluster.inv_mass_central<=0.0f||cluster.inv_mass_h<=0.0f)return false;
    for(int h=0;h<cluster.n_hydrogens;++h) {
        const int atom=cluster.hydrogen_atoms[h];if(atom<0)continue;
        const float length=cluster.bond_lengths[h];
        if(!isfinite(length)||length<=0.0f)return false;
        for(int iteration=0;iteration<10;++iteration) {
            float3& central=x[cluster.central_atom];float3& hydrogen=x[atom];
            const float3 r=make_float3(hydrogen.x-central.x,hydrogen.y-central.y,hydrogen.z-central.z);
            const float square=r.x*r.x+r.y*r.y+r.z*r.z;
            if(!isfinite(square)||square<=0.0f)return false;
            const float target=length*length;const float difference=square-target;
            if(fabsf(difference)<1.0e-6f)break;
            const float inverse=cluster.inv_mass_central+cluster.inv_mass_h;
            const float factor=difference/(2.0f*inverse*square);
            const float3 correction=make_float3(factor*r.x,factor*r.y,factor*r.z);
            central.x+=cluster.inv_mass_central*correction.x;
            central.y+=cluster.inv_mass_central*correction.y;
            central.z+=cluster.inv_mass_central*correction.z;
            hydrogen.x-=cluster.inv_mass_h*correction.x;
            hydrogen.y-=cluster.inv_mass_h*correction.y;
            hydrogen.z-=cluster.inv_mass_h*correction.z;
            if(!isfinite(central.x)||!isfinite(central.y)||!isfinite(central.z)||
               !isfinite(hydrogen.x)||!isfinite(hydrogen.y)||!isfinite(hydrogen.z))return false;
        }
    }
    return true;
}

template<bool Compensated=false>
__device__ bool shake_cluster_positions(
    float3* positions,
    const float3* positions_step_start,
    float3* velocities,
    const HCluster cluster,
    float dt,
    double* nonham_work,
    int max_iter,
    int diagnostic_native_step
,
    const BiprismCoordinateOwnerFfi* coordinate_owner=nullptr,
    uint32_t* precision_fault=nullptr
) {
    if constexpr(!Compensated) {
    // The coordinate state is f32 and protein coordinates are commonly O(100 A).
    // An absolute 1e-6 A^2 criterion is below the representable correction at
    // that scale and deterministically stalls otherwise valid constraints.
    // Evaluate a dimensionless squared-distance residual instead.
    const float relative_tolerance = 1.0e-4f;
    const float3 central_velocity_before = velocities[cluster.central_atom];
    const float3 central_position_before = positions[cluster.central_atom];
    float3 hydrogen_positions_before[3];
    float3 hydrogen_velocities_before[3];
    #pragma unroll
    for (int h = 0; h < 3; h++) {
        const int h_atom = cluster.hydrogen_atoms[h];
        hydrogen_positions_before[h] = h_atom >= 0 ? positions[h_atom] : make_float3(0, 0, 0);
        hydrogen_velocities_before[h] = h_atom >= 0 ? velocities[h_atom] : make_float3(0, 0, 0);
    }
    bool converged = false;
    for (int iter = 0; iter < max_iter; iter++) {
        float max_relative_error = 0.0f;
        for (int h = 0; h < cluster.n_hydrogens; h++) {
            const int h_atom = cluster.hydrogen_atoms[h];
            if (h_atom < 0) continue;
            float3& pos_central = positions[cluster.central_atom];
            float3& pos_h = positions[h_atom];
            const float3 r = make_float3(
                pos_h.x - pos_central.x,
                pos_h.y - pos_central.y,
                pos_h.z - pos_central.z);
            const float3 r0 = make_float3(
                positions_step_start[h_atom].x - positions_step_start[cluster.central_atom].x,
                positions_step_start[h_atom].y - positions_step_start[cluster.central_atom].y,
                positions_step_start[h_atom].z - positions_step_start[cluster.central_atom].z);
            const float r_sq = r.x*r.x + r.y*r.y + r.z*r.z;
            const float target_sq = cluster.bond_lengths[h] * cluster.bond_lengths[h];
            const float diff = r_sq - target_sq;
            if (!isfinite(r_sq) || !isfinite(target_sq) || target_sq <= 0.0f) {
                printf("PRISM_SHAKE_INVALID_LENGTH central=%d hydrogen=%d r2=%.9g target2=%.9g dt=%.9g\n",
                    cluster.central_atom,h_atom,(double)r_sq,(double)target_sq,(double)dt);
                return false;
            }
            const float relative_error = fabsf(diff) / target_sq;
            max_relative_error = fmaxf(max_relative_error, relative_error);
            if (relative_error <= relative_tolerance) continue;
            const float r_dot_r0 = r.x*r0.x + r.y*r0.y + r.z*r0.z;
            const float r0_sq = r0.x*r0.x + r0.y*r0.y + r0.z*r0.z;
            const float direction_scale = sqrtf(r_sq * r0_sq);
            if (!isfinite(r_dot_r0) || !isfinite(direction_scale)
                    || direction_scale <= 0.0f
                    || fabsf(r_dot_r0) / direction_scale <= 1.0e-6f
                    || !isfinite(dt) || dt <= 0.0f) {
                printf("PRISM_SHAKE_INVALID_DIRECTION central=%d hydrogen=%d r2=%.9g dot=%.9g scale=%.9g dt=%.9g\n",
                    cluster.central_atom,h_atom,(double)r_sq,(double)r_dot_r0,(double)direction_scale,(double)dt);
                return false;
            }
            const float inv_mass_sum = cluster.inv_mass_central + cluster.inv_mass_h;
            const float lambda = diff / (2.0f * inv_mass_sum * r_dot_r0);
            const float3 correction = make_float3(lambda * r0.x, lambda * r0.y, lambda * r0.z);
            pos_central.x += cluster.inv_mass_central * correction.x;
            pos_central.y += cluster.inv_mass_central * correction.y;
            pos_central.z += cluster.inv_mass_central * correction.z;
            pos_h.x -= cluster.inv_mass_h * correction.x;
            pos_h.y -= cluster.inv_mass_h * correction.y;
            pos_h.z -= cluster.inv_mass_h * correction.z;
            velocities[cluster.central_atom].x += cluster.inv_mass_central * correction.x / dt;
            velocities[cluster.central_atom].y += cluster.inv_mass_central * correction.y / dt;
            velocities[cluster.central_atom].z += cluster.inv_mass_central * correction.z / dt;
            velocities[h_atom].x -= cluster.inv_mass_h * correction.x / dt;
            velocities[h_atom].y -= cluster.inv_mass_h * correction.y / dt;
            velocities[h_atom].z -= cluster.inv_mass_h * correction.z / dt;
        }
        if (max_relative_error <= relative_tolerance) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        const float3 center=positions[cluster.central_atom];
        for(int h=0;h<cluster.n_hydrogens;++h) {
            const int atom=cluster.hydrogen_atoms[h];if(atom<0)continue;
            const float3 point=positions[atom];
            const double dx=(double)point.x-center.x,dy=(double)point.y-center.y,dz=(double)point.z-center.z;
            const double target=(double)cluster.bond_lengths[h]*cluster.bond_lengths[h];
            // Failure-only diagnostic. Initial drift and old-bond geometry determine
            // whether the unchanged SHAKE correction direction intersects the sphere.
            const float3 initial_h = hydrogen_positions_before[h];
            const double rx = (double)initial_h.x - central_position_before.x;
            const double ry = (double)initial_h.y - central_position_before.y;
            const double rz = (double)initial_h.z - central_position_before.z;
            const double ox = (double)positions_step_start[atom].x - positions_step_start[cluster.central_atom].x;
            const double oy = (double)positions_step_start[atom].y - positions_step_start[cluster.central_atom].y;
            const double oz = (double)positions_step_start[atom].z - positions_step_start[cluster.central_atom].z;
            const double old_sq = ox*ox + oy*oy + oz*oz;
            const double initial_sq = rx*rx + ry*ry + rz*rz;
            const double initial_dot_old = rx*ox + ry*oy + rz*oz;
            const double perpendicular_sq = old_sq > 0.0
                ? initial_sq - initial_dot_old*initial_dot_old/old_sq : NAN;
            const double discriminant = initial_dot_old*initial_dot_old - old_sq*(initial_sq-target);
            const float3 initial_vh = hydrogen_velocities_before[h];
            printf("PRISM_SHAKE_INITIAL_STATE_V1 step=%d central=%d hydrogen=%d iterations=%d dt=%.9g target=%.9g inv_mass_c=%.9g inv_mass_h=%.9g r=(%.17g,%.17g,%.17g) r0=(%.17g,%.17g,%.17g) vc=(%.9g,%.9g,%.9g) vh=(%.9g,%.9g,%.9g) perpendicular_sq=%.17g discriminant=%.17g source=nhs_amber_fused.cu:shake_cluster_positions\n",
                diagnostic_native_step,cluster.central_atom,atom,max_iter,(double)dt,
                (double)cluster.bond_lengths[h],(double)cluster.inv_mass_central,(double)cluster.inv_mass_h,
                rx,ry,rz,ox,oy,oz,
                (double)central_velocity_before.x,(double)central_velocity_before.y,(double)central_velocity_before.z,
                (double)initial_vh.x,(double)initial_vh.y,(double)initial_vh.z,perpendicular_sq,discriminant);
            printf("PRISM_SHAKE_NONCONVERGENCE central=%d hydrogen=%d iterations=%d dt=%.9g relative_residual=%.17g center=(%.9g,%.9g,%.9g) hydrogen_xyz=(%.9g,%.9g,%.9g)\n",
                cluster.central_atom,atom,max_iter,(double)dt,fabs(dx*dx+dy*dy+dz*dz-target)/target,
                (double)center.x,(double)center.y,(double)center.z,(double)point.x,(double)point.y,(double)point.z);
        }
        return false;
    }
    if (nonham_work != nullptr) {
        const float3 central_after = velocities[cluster.central_atom];
        const double central_before_sq = (double)central_velocity_before.x * central_velocity_before.x
                                       + (double)central_velocity_before.y * central_velocity_before.y
                                       + (double)central_velocity_before.z * central_velocity_before.z;
        const double central_after_sq = (double)central_after.x * central_after.x
                                      + (double)central_after.y * central_after.y
                                      + (double)central_after.z * central_after.z;
        nonham_work[cluster.central_atom] += 0.5 * (1.0 / (double)cluster.inv_mass_central)
            * (central_after_sq - central_before_sq) / (double)AMBER_KCAL_TO_AMU_A2_PS2;
        for (int h = 0; h < cluster.n_hydrogens; h++) {
            const int h_atom = cluster.hydrogen_atoms[h];
            if (h_atom < 0) continue;
            const float3 after = velocities[h_atom];
            const float3 before = hydrogen_velocities_before[h];
            const double before_sq = (double)before.x * before.x + (double)before.y * before.y
                                   + (double)before.z * before.z;
            const double after_sq = (double)after.x * after.x + (double)after.y * after.y
                                  + (double)after.z * after.z;
            nonham_work[h_atom] += 0.5 * (1.0 / (double)cluster.inv_mass_h)
                * (after_sq - before_sq) / (double)AMBER_KCAL_TO_AMU_A2_PS2;
        }
    }
    return true;

    } else {
    using CG=BpNativeGeometry<Compensated>;
    using CReal=typename CG::Real;
    using CVec=typename CG::Vec;
    // The coordinate state is f32 and protein coordinates are commonly O(100 A).
    // An absolute 1e-6 A^2 criterion is below the representable correction at
    // that scale and deterministically stalls otherwise valid constraints.
    // Evaluate a dimensionless squared-distance residual instead.
    const float relative_tolerance = 1.0e-4f;
    const float3 central_velocity_before = velocities[cluster.central_atom];
    const CVec central_position_before = CG::load(positions,cluster.central_atom,coordinate_owner,precision_fault);
    CVec hydrogen_positions_before[3];
    float3 hydrogen_velocities_before[3];
    #pragma unroll
    for (int h = 0; h < 3; h++) {
        const int h_atom = cluster.hydrogen_atoms[h];
        hydrogen_positions_before[h] = h_atom >= 0 ? CG::load(positions,h_atom,coordinate_owner,precision_fault) : CG::make(0, 0, 0);
        hydrogen_velocities_before[h] = h_atom >= 0 ? velocities[h_atom] : make_float3(0, 0, 0);
    }
    bool converged = false;
    for (int iter = 0; iter < max_iter; iter++) {
        CReal max_relative_error = 0.0f;
        for (int h = 0; h < cluster.n_hydrogens; h++) {
            const int h_atom = cluster.hydrogen_atoms[h];
            if (h_atom < 0) continue;
            const CVec r=CG::difference(positions,h_atom,cluster.central_atom,
                coordinate_owner,BiprismCoordinatePlane::Current,precision_fault);
            const CVec r0=CG::difference(positions_step_start,h_atom,cluster.central_atom,
                coordinate_owner,BiprismCoordinatePlane::StepStart,precision_fault);
            const CReal r_sq = r.x*r.x + r.y*r.y + r.z*r.z;
            const CReal target_sq = (CReal)cluster.bond_lengths[h] * cluster.bond_lengths[h];
            const CReal diff = r_sq - target_sq;
            if (!isfinite(r_sq) || !isfinite(target_sq) || target_sq <= 0.0f) {
                printf("PRISM_SHAKE_INVALID_LENGTH central=%d hydrogen=%d r2=%.9g target2=%.9g dt=%.9g\n",
                    cluster.central_atom,h_atom,(double)r_sq,(double)target_sq,(double)dt);
                return false;
            }
            const CReal relative_error = bp_native_abs(diff) / target_sq;
            max_relative_error = bp_native_max(max_relative_error, relative_error);
            if (relative_error <= relative_tolerance) continue;
            const CReal r_dot_r0 = r.x*r0.x + r.y*r0.y + r.z*r0.z;
            const CReal r0_sq = r0.x*r0.x + r0.y*r0.y + r0.z*r0.z;
            const CReal direction_scale = bp_native_sqrt(r_sq * r0_sq);
            if (!isfinite(r_dot_r0) || !isfinite(direction_scale)
                    || direction_scale <= 0.0f
                    || bp_native_abs(r_dot_r0) / direction_scale <= 1.0e-6f
                    || !isfinite(dt) || dt <= 0.0f) {
                printf("PRISM_SHAKE_INVALID_DIRECTION central=%d hydrogen=%d r2=%.9g dot=%.9g scale=%.9g dt=%.9g\n",
                    cluster.central_atom,h_atom,(double)r_sq,(double)r_dot_r0,(double)direction_scale,(double)dt);
                return false;
            }
            const CReal inv_mass_sum = (CReal)cluster.inv_mass_central + cluster.inv_mass_h;
            const CReal lambda = diff / (2.0f * inv_mass_sum * r_dot_r0);
            const CVec correction = CG::make(lambda * r0.x, lambda * r0.y, lambda * r0.z);
            if(!CG::add_current(positions,cluster.central_atom,CG::make(
                    cluster.inv_mass_central*correction.x,cluster.inv_mass_central*correction.y,
                    cluster.inv_mass_central*correction.z),coordinate_owner,precision_fault) ||
               !CG::add_current(positions,h_atom,CG::make(
                    -cluster.inv_mass_h*correction.x,-cluster.inv_mass_h*correction.y,
                    -cluster.inv_mass_h*correction.z),coordinate_owner,precision_fault))return false;
            velocities[cluster.central_atom].x += cluster.inv_mass_central * correction.x / dt;
            velocities[cluster.central_atom].y += cluster.inv_mass_central * correction.y / dt;
            velocities[cluster.central_atom].z += cluster.inv_mass_central * correction.z / dt;
            velocities[h_atom].x -= cluster.inv_mass_h * correction.x / dt;
            velocities[h_atom].y -= cluster.inv_mass_h * correction.y / dt;
            velocities[h_atom].z -= cluster.inv_mass_h * correction.z / dt;
        }
        if (max_relative_error <= relative_tolerance) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        const CVec center=CG::load(positions,cluster.central_atom,coordinate_owner,precision_fault);
        for(int h=0;h<cluster.n_hydrogens;++h) {
            const int atom=cluster.hydrogen_atoms[h];if(atom<0)continue;
            const CVec point=CG::load(positions,atom,coordinate_owner,precision_fault);
            const double dx=(double)point.x-center.x,dy=(double)point.y-center.y,dz=(double)point.z-center.z;
            const double target=(double)cluster.bond_lengths[h]*cluster.bond_lengths[h];
            // Failure-only diagnostic. Initial drift and old-bond geometry determine
            // whether the unchanged SHAKE correction direction intersects the sphere.
            const CVec initial_h = hydrogen_positions_before[h];
            const double rx = (double)initial_h.x - central_position_before.x;
            const double ry = (double)initial_h.y - central_position_before.y;
            const double rz = (double)initial_h.z - central_position_before.z;
            const double ox = (double)CG::load_plane(positions_step_start,atom,coordinate_owner,BiprismCoordinatePlane::StepStart,precision_fault).x - CG::load_plane(positions_step_start,cluster.central_atom,coordinate_owner,BiprismCoordinatePlane::StepStart,precision_fault).x;
            const double oy = (double)CG::load_plane(positions_step_start,atom,coordinate_owner,BiprismCoordinatePlane::StepStart,precision_fault).y - CG::load_plane(positions_step_start,cluster.central_atom,coordinate_owner,BiprismCoordinatePlane::StepStart,precision_fault).y;
            const double oz = (double)CG::load_plane(positions_step_start,atom,coordinate_owner,BiprismCoordinatePlane::StepStart,precision_fault).z - CG::load_plane(positions_step_start,cluster.central_atom,coordinate_owner,BiprismCoordinatePlane::StepStart,precision_fault).z;
            const double old_sq = ox*ox + oy*oy + oz*oz;
            const double initial_sq = rx*rx + ry*ry + rz*rz;
            const double initial_dot_old = rx*ox + ry*oy + rz*oz;
            const double perpendicular_sq = old_sq > 0.0
                ? initial_sq - initial_dot_old*initial_dot_old/old_sq : NAN;
            const double discriminant = initial_dot_old*initial_dot_old - old_sq*(initial_sq-target);
            const float3 initial_vh = hydrogen_velocities_before[h];
            printf("PRISM_SHAKE_INITIAL_STATE_V1 step=%d central=%d hydrogen=%d iterations=%d dt=%.9g target=%.9g inv_mass_c=%.9g inv_mass_h=%.9g r=(%.17g,%.17g,%.17g) r0=(%.17g,%.17g,%.17g) vc=(%.9g,%.9g,%.9g) vh=(%.9g,%.9g,%.9g) perpendicular_sq=%.17g discriminant=%.17g source=nhs_amber_fused.cu:shake_cluster_positions\n",
                diagnostic_native_step,cluster.central_atom,atom,max_iter,(double)dt,
                (double)cluster.bond_lengths[h],(double)cluster.inv_mass_central,(double)cluster.inv_mass_h,
                rx,ry,rz,ox,oy,oz,
                (double)central_velocity_before.x,(double)central_velocity_before.y,(double)central_velocity_before.z,
                (double)initial_vh.x,(double)initial_vh.y,(double)initial_vh.z,perpendicular_sq,discriminant);
            printf("PRISM_SHAKE_NONCONVERGENCE central=%d hydrogen=%d iterations=%d dt=%.9g relative_residual=%.17g center=(%.9g,%.9g,%.9g) hydrogen_xyz=(%.9g,%.9g,%.9g)\n",
                cluster.central_atom,atom,max_iter,(double)dt,fabs(dx*dx+dy*dy+dz*dz-target)/target,
                (double)center.x,(double)center.y,(double)center.z,(double)point.x,(double)point.y,(double)point.z);
        }
        return false;
    }
    if (nonham_work != nullptr) {
        const float3 central_after = velocities[cluster.central_atom];
        const double central_before_sq = (double)central_velocity_before.x * central_velocity_before.x
                                       + (double)central_velocity_before.y * central_velocity_before.y
                                       + (double)central_velocity_before.z * central_velocity_before.z;
        const double central_after_sq = (double)central_after.x * central_after.x
                                      + (double)central_after.y * central_after.y
                                      + (double)central_after.z * central_after.z;
        nonham_work[cluster.central_atom] += 0.5 * (1.0 / (double)cluster.inv_mass_central)
            * (central_after_sq - central_before_sq) / (double)AMBER_KCAL_TO_AMU_A2_PS2;
        for (int h = 0; h < cluster.n_hydrogens; h++) {
            const int h_atom = cluster.hydrogen_atoms[h];
            if (h_atom < 0) continue;
            const float3 after = velocities[h_atom];
            const float3 before = hydrogen_velocities_before[h];
            const double before_sq = (double)before.x * before.x + (double)before.y * before.y
                                   + (double)before.z * before.z;
            const double after_sq = (double)after.x * after.x + (double)after.y * after.y
                                  + (double)after.z * after.z;
            nonham_work[h_atom] += 0.5 * (1.0 / (double)cluster.inv_mass_h)
                * (after_sq - before_sq) / (double)AMBER_KCAL_TO_AMU_A2_PS2;
        }
    }
    return true;

    }
}

template<bool Compensated=false>
__device__ bool rattle_cluster_velocities(
    const float3* positions,
    float3* velocities,
    const HCluster cluster,
    double* nonham_work,
    int max_iter
,
    const BiprismCoordinateOwnerFfi* coordinate_owner=nullptr,
    uint32_t* precision_fault=nullptr
) {
    if constexpr(!Compensated) {
    // Normalize the velocity constraint by bond length and relative-speed
    // scale.  This keeps the convergence criterion dimensionless and
    // attainable in f32 after force kicks.
    const float relative_tolerance = 1.0e-5f;
    const float3 central_velocity_before = velocities[cluster.central_atom];
    float3 hydrogen_velocities_before[3];
    #pragma unroll
    for (int h = 0; h < 3; h++) {
        const int h_atom = cluster.hydrogen_atoms[h];
        hydrogen_velocities_before[h] = h_atom >= 0 ? velocities[h_atom] : make_float3(0, 0, 0);
    }
    bool converged = false;
    for (int iter = 0; iter < max_iter; iter++) {
        float max_relative_error = 0.0f;
        for (int h = 0; h < cluster.n_hydrogens; h++) {
            const int h_atom = cluster.hydrogen_atoms[h];
            if (h_atom < 0) continue;
            const float3 r = make_float3(
                positions[h_atom].x - positions[cluster.central_atom].x,
                positions[h_atom].y - positions[cluster.central_atom].y,
                positions[h_atom].z - positions[cluster.central_atom].z);
            const float r_sq = r.x*r.x + r.y*r.y + r.z*r.z;
            if (!isfinite(r_sq) || r_sq < 1.0e-12f) return false;
            float3& vel_central = velocities[cluster.central_atom];
            float3& vel_h = velocities[h_atom];
            const float3 relative_velocity = make_float3(
                vel_h.x - vel_central.x,
                vel_h.y - vel_central.y,
                vel_h.z - vel_central.z);
            const float residual = r.x * relative_velocity.x
                                 + r.y * relative_velocity.y
                                 + r.z * relative_velocity.z;
            const float relative_speed = sqrtf(
                relative_velocity.x * relative_velocity.x
              + relative_velocity.y * relative_velocity.y
              + relative_velocity.z * relative_velocity.z);
            const float residual_scale = sqrtf(r_sq) * fmaxf(relative_speed, 1.0f);
            const float relative_error = fabsf(residual) / residual_scale;
            if (!isfinite(relative_error)) return false;
            max_relative_error = fmaxf(max_relative_error, relative_error);
            if (relative_error <= relative_tolerance) continue;
            const float inv_mass_sum = cluster.inv_mass_central + cluster.inv_mass_h;
            const float lambda = residual / (inv_mass_sum * r_sq);
            const float3 correction = make_float3(lambda * r.x, lambda * r.y, lambda * r.z);
            vel_central.x += cluster.inv_mass_central * correction.x;
            vel_central.y += cluster.inv_mass_central * correction.y;
            vel_central.z += cluster.inv_mass_central * correction.z;
            vel_h.x -= cluster.inv_mass_h * correction.x;
            vel_h.y -= cluster.inv_mass_h * correction.y;
            vel_h.z -= cluster.inv_mass_h * correction.z;
        }
        if (max_relative_error <= relative_tolerance) {
            converged = true;
            break;
        }
    }

    if (!converged) return false;
    if (nonham_work != nullptr) {
        const float3 central_after = velocities[cluster.central_atom];
        const double central_before_sq = (double)central_velocity_before.x * central_velocity_before.x
                                       + (double)central_velocity_before.y * central_velocity_before.y
                                       + (double)central_velocity_before.z * central_velocity_before.z;
        const double central_after_sq = (double)central_after.x * central_after.x
                                      + (double)central_after.y * central_after.y
                                      + (double)central_after.z * central_after.z;
        nonham_work[cluster.central_atom] += 0.5 * (1.0 / (double)cluster.inv_mass_central)
            * (central_after_sq - central_before_sq) / (double)AMBER_KCAL_TO_AMU_A2_PS2;
        for (int h = 0; h < cluster.n_hydrogens; h++) {
            const int h_atom = cluster.hydrogen_atoms[h];
            if (h_atom < 0) continue;
            const float3 after = velocities[h_atom];
            const float3 before = hydrogen_velocities_before[h];
            const double before_sq = (double)before.x * before.x + (double)before.y * before.y
                                   + (double)before.z * before.z;
            const double after_sq = (double)after.x * after.x + (double)after.y * after.y
                                  + (double)after.z * after.z;
            nonham_work[h_atom] += 0.5 * (1.0 / (double)cluster.inv_mass_h)
                * (after_sq - before_sq) / (double)AMBER_KCAL_TO_AMU_A2_PS2;
        }
    }
    return true;

    } else {
    using CG=BpNativeGeometry<Compensated>;
    using CReal=typename CG::Real;
    using CVec=typename CG::Vec;
    // Normalize the velocity constraint by bond length and relative-speed
    // scale.  This keeps the convergence criterion dimensionless and
    // attainable in f32 after force kicks.
    const float relative_tolerance = 1.0e-5f;
    const float3 central_velocity_before = velocities[cluster.central_atom];
    float3 hydrogen_velocities_before[3];
    #pragma unroll
    for (int h = 0; h < 3; h++) {
        const int h_atom = cluster.hydrogen_atoms[h];
        hydrogen_velocities_before[h] = h_atom >= 0 ? velocities[h_atom] : make_float3(0, 0, 0);
    }
    bool converged = false;
    for (int iter = 0; iter < max_iter; iter++) {
        CReal max_relative_error = 0.0f;
        for (int h = 0; h < cluster.n_hydrogens; h++) {
            const int h_atom = cluster.hydrogen_atoms[h];
            if (h_atom < 0) continue;
            const CVec r=CG::difference(positions,h_atom,cluster.central_atom,
                coordinate_owner,BiprismCoordinatePlane::Current,precision_fault);
            const CReal r_sq = r.x*r.x + r.y*r.y + r.z*r.z;
            if (!isfinite(r_sq) || r_sq < 1.0e-12f) return false;
            float3& vel_central = velocities[cluster.central_atom];
            float3& vel_h = velocities[h_atom];
            const CVec relative_velocity = CG::make(
                (CReal)vel_h.x - vel_central.x,
                (CReal)vel_h.y - vel_central.y,
                (CReal)vel_h.z - vel_central.z);
            const CReal residual = r.x * relative_velocity.x
                                 + r.y * relative_velocity.y
                                 + r.z * relative_velocity.z;
            const CReal relative_speed = bp_native_sqrt(
                relative_velocity.x * relative_velocity.x
              + relative_velocity.y * relative_velocity.y
              + relative_velocity.z * relative_velocity.z);
            const CReal residual_scale = bp_native_sqrt(r_sq) * bp_native_max(relative_speed, 1.0f);
            const CReal relative_error = bp_native_abs(residual) / residual_scale;
            if (!isfinite(relative_error)) return false;
            max_relative_error = bp_native_max(max_relative_error, relative_error);
            if (relative_error <= relative_tolerance) continue;
            const CReal inv_mass_sum = (CReal)cluster.inv_mass_central + cluster.inv_mass_h;
            const CReal lambda = residual / (inv_mass_sum * r_sq);
            const CVec correction = CG::make(lambda * r.x, lambda * r.y, lambda * r.z);
            vel_central.x += cluster.inv_mass_central * correction.x;
            vel_central.y += cluster.inv_mass_central * correction.y;
            vel_central.z += cluster.inv_mass_central * correction.z;
            vel_h.x -= cluster.inv_mass_h * correction.x;
            vel_h.y -= cluster.inv_mass_h * correction.y;
            vel_h.z -= cluster.inv_mass_h * correction.z;
        }
        if (max_relative_error <= relative_tolerance) {
            converged = true;
            break;
        }
    }

    if (!converged) return false;
    if (nonham_work != nullptr) {
        const float3 central_after = velocities[cluster.central_atom];
        const double central_before_sq = (double)central_velocity_before.x * central_velocity_before.x
                                       + (double)central_velocity_before.y * central_velocity_before.y
                                       + (double)central_velocity_before.z * central_velocity_before.z;
        const double central_after_sq = (double)central_after.x * central_after.x
                                      + (double)central_after.y * central_after.y
                                      + (double)central_after.z * central_after.z;
        nonham_work[cluster.central_atom] += 0.5 * (1.0 / (double)cluster.inv_mass_central)
            * (central_after_sq - central_before_sq) / (double)AMBER_KCAL_TO_AMU_A2_PS2;
        for (int h = 0; h < cluster.n_hydrogens; h++) {
            const int h_atom = cluster.hydrogen_atoms[h];
            if (h_atom < 0) continue;
            const float3 after = velocities[h_atom];
            const float3 before = hydrogen_velocities_before[h];
            const double before_sq = (double)before.x * before.x + (double)before.y * before.y
                                   + (double)before.z * before.z;
            const double after_sq = (double)after.x * after.x + (double)after.y * after.y
                                  + (double)after.z * after.z;
            nonham_work[h_atom] += 0.5 * (1.0 / (double)cluster.inv_mass_h)
                * (after_sq - before_sq) / (double)AMBER_KCAL_TO_AMU_A2_PS2;
        }
    }
    return true;

    }
}

// ============================================================================
// HOLOGRAPHIC EXCLUSION FIELD
// ============================================================================

// Atom type radii for exclusion field
__device__ float get_exclusion_radius(int atom_type) {
    // 0=hydrophobic, 1=polar, 2=charged+, 3=charged-, 4=aromatic, 5=backbone
    const float radii[] = {4.5f, 2.5f, 2.0f, 2.0f, 3.5f, 2.0f, 1.4f};
    return radii[atom_type % 7];
}

// Compute exclusion contribution from single atom to voxel
__device__ float compute_exclusion_contribution(
    float3 atom_pos,
    float3 voxel_center,
    int atom_type,
    float charge
) {
    float3 d = make_float3(
        voxel_center.x - atom_pos.x,
        voxel_center.y - atom_pos.y,
        voxel_center.z - atom_pos.z
    );
    float dist_sq = d.x*d.x + d.y*d.y + d.z*d.z;
    float dist = sqrtf(dist_sq);

    float radius = get_exclusion_radius(atom_type);

    // Gaussian exclusion
    float sigma = radius / 2.0f;
    float exclusion = expf(-dist_sq / (2.0f * sigma * sigma));

    // Charge modulation (charged atoms have stronger exclusion)
    float charge_factor = 1.0f + 0.5f * fabsf(charge);

    return exclusion * charge_factor;
}

// Infer water density from exclusion field
__device__ float infer_water_density(
    float exclusion_value,
    float polar_field,
    float temperature
) {
    // Base density reduced by exclusion
    float base_density = WATER_DENSITY_BULK * (1.0f - exclusion_value);

    // Polar enhancement (water clusters near polar groups)
    float polar_enhancement = 1.0f + 0.3f * polar_field;

    // Temperature effect (colder = less mobile water, clearer signal)
    // At low temps, water behavior becomes more discrete (freeze transition)
    float temp_factor = sqrtf(temperature / 300.0f);

    return base_density * polar_enhancement * temp_factor;
}

// ============================================================================
// NEUROMORPHIC LIF OBSERVATION
// ============================================================================

// LIF neuron update - BIDIRECTIONAL detection (rewetting AND dewetting)
// For cryo-thermal detection, we detect BOTH:
// 1. Dewetting: water pushed away (exclusion increase)
// 2. Rewetting: water attracted (exclusion decrease from UV excitation)
//
// Returns: true if spike occurred
// Output: spike_intensity is set to the membrane potential that triggered the spike
__device__ bool lif_neuron_update(
    float& membrane_potential,
    float water_density_current,
    float water_density_prev,
    float tau_mem,
    float dt,
    float threshold,
    float& spike_intensity  // OUTPUT: intensity of spike (0 if no spike)
) {
    spike_intensity = 0.0f;

    // Bidirectional input: detect changes above noise floor
    // TUNED for Cryo-UV: sensitive to dewetting events during temperature ramp
    float density_change = water_density_current - water_density_prev;

    // Lower noise floor for cryo-UV sensitivity (thermal fluctuation ~0.001)
    const float NOISE_FLOOR = 0.002f;
    float abs_change = fabsf(density_change);
    // Increased amplification (10x) for cryo-UV detection
    float bidirectional_signal = (abs_change > NOISE_FLOOR) ? (abs_change - NOISE_FLOOR) * 3.0f : 0.0f;

    // Exclusion-weighted term: detect hydrophobic dewetting zones
    float density_deviation = fabsf(water_density_current - WATER_DENSITY_BULK);
    const float EXCLUSION_NOISE_FLOOR = 0.005f;  // Lower floor for cryptic sites
    // Increased weight (5x) for exclusion-based detection
    float exclusion_signal = (density_deviation > EXCLUSION_NOISE_FLOOR) ?
        (density_deviation - EXCLUSION_NOISE_FLOOR) * 2.0f : 0.0f;

    float combined_signal = bidirectional_signal + exclusion_signal;

    // Slower decay to allow signal accumulation at cryptic sites
    float effective_tau = tau_mem * 1.5f;  // Slower decay for cryptic site detection
    float decay = expf(-dt / effective_tau);
    membrane_potential = decay * membrane_potential + combined_signal;

    // Spike check - threshold tuned for differential detection
    bool spike = membrane_potential >= threshold;
    if (spike) {
        // Capture the intensity BEFORE resetting
        spike_intensity = membrane_potential;
        membrane_potential = LIF_RESET;
    }

    return spike;
}

// ============================================================================
// UV BIAS PUMP-PROBE
// ============================================================================

// Check if atom is UV-absorbing aromatic
__device__ bool is_uv_absorber(int atom_type, int residue_type) {
    // atom_type 4 = aromatic
    // residue_type: TRP=0, TYR=1, PHE=2
    return atom_type == 4;
}

// Apply UV burst energy to aromatic atoms
__device__ void apply_uv_burst(
    float3& velocity,
    float mass,
    float absorption_strength,  // TRP=1.0, TYR=0.5, PHE=0.2
    float burst_energy,
    curandState* rng_state
) {
    // Convert energy to velocity perturbation
    // E = 0.5 * m * v^2 => v = sqrt(2*E/m)
    float energy_absorbed = burst_energy * absorption_strength;
    float velocity_boost = sqrtf(2.0f * energy_absorbed / mass);

    // Random direction for energy deposition
    float theta = 2.0f * PI * curand_uniform(rng_state);
    float phi = acosf(2.0f * curand_uniform(rng_state) - 1.0f);

    velocity.x += velocity_boost * sinf(phi) * cosf(theta);
    velocity.y += velocity_boost * sinf(phi) * sinf(theta);
    velocity.z += velocity_boost * cosf(phi);
}

// ============================================================================
// WARP MATRIX - VOXEL TO ATOM MAPPING
// ============================================================================

// Build warp matrix entry for a voxel
__device__ void build_warp_entry(
    WarpEntry& entry,
    float3 voxel_center,
    const float3* positions,
    int n_atoms,
    float cutoff
) {
    entry.voxel_idx = -1;
    entry.n_atoms = 0;

    float total_weight = 0.0f;

    for (int i = 0; i < n_atoms && entry.n_atoms < 16; i++) {
        float3 d = make_float3(
            positions[i].x - voxel_center.x,
            positions[i].y - voxel_center.y,
            positions[i].z - voxel_center.z
        );
        float dist = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);

        if (dist < cutoff) {
            float weight = 1.0f / (dist + 0.1f);  // Inverse distance weight
            entry.atom_indices[entry.n_atoms] = i;
            entry.atom_weights[entry.n_atoms] = weight;
            total_weight += weight;
            entry.n_atoms++;
        }
    }

    // Normalize weights
    if (total_weight > 0.0f) {
        for (int i = 0; i < entry.n_atoms; i++) {
            entry.atom_weights[i] /= total_weight;
        }
    }
}

// ============================================================================
// SIGNAL PRESERVATION: per-spike update of voxel-level tracking buffers
// ============================================================================

__device__ __forceinline__ void update_signal_preservation(
    int v, int timestep, int spike_source,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    const WarpEntry& entry,
    const int* residue_ids,
    int n_atoms,
    int* residue_step_causal  // KCC: per-residue per-step causal counter (may be nullptr)
) {
    // 1. Spatial recurrence
    atomicAdd(&voxel_hit_grid[v], 1u);

    // 2. UV→LIF causal linkage
    if (spike_source == 1) {
        // UV spike: record timestamp for downstream causal check
        last_uv_step[v] = timestep;
    } else {
        // LIF/RAF/EFP spike: check for recent UV activity at this voxel
        int last_uv = last_uv_step[v];
        if (last_uv >= 0 && (timestep - last_uv) < UV_LIF_WINDOW) {
            atomicAdd(&coupled_spike_grid[v], 1u);
            // KCC: increment per-residue causal counter for the driving residue
            if (residue_step_causal != nullptr && entry.n_atoms > 0) {
                int a0 = entry.atom_indices[0];
                if (a0 >= 0 && a0 < n_atoms) {
                    int res_id = residue_ids[a0];
                    if (res_id >= 0) {
                        atomicAdd(&residue_step_causal[res_id], 1);
                    }
                }
            }
        }
    }

    // 3. Dominant residue tracking (top-1 Misra-Gries heavy hitter)
    int primary_res = -1;
    if (entry.n_atoms > 0) {
        int a0 = entry.atom_indices[0];  // highest-weight contributing atom
        if (a0 >= 0 && a0 < n_atoms) {
            primary_res = residue_ids[a0];
        }
    }
    if (primary_res >= 0) {
        int stored = primary_residue_id[v];
        if (stored < 0) {
            // First spike at this voxel
            primary_residue_id[v] = primary_res;
            primary_residue_count[v] = 1u;
        } else if (stored == primary_res) {
            primary_residue_count[v]++;
        } else {
            // Different residue: Misra-Gries decrement-or-replace
            unsigned int cnt = primary_residue_count[v];
            if (cnt <= 1u) {
                primary_residue_id[v] = primary_res;
                primary_residue_count[v] = 1u;
            } else {
                primary_residue_count[v] = cnt - 1u;
            }
        }
    }
}

// ============================================================================
// SPIKE-TRIGGERED SNAPSHOT CAPTURE
// ============================================================================

__device__ void capture_spike_event(
    SpikeEvent& event,
    int timestep,
    int voxel_idx,
    float3 voxel_center,
    float intensity,
    const WarpEntry& warp_entry,
    const int* residue_ids,
    int spike_source,            // 1=UV, 2=LIF
    float wavelength_nm,         // UV wavelength (0 for LIF)
    int aromatic_type,           // 0=TRP,1=TYR,2=PHE,3=SS,-1=none
    int aromatic_residue_id,     // closest excited aromatic residue (-1 for LIF)
    float water_density,         // local water density
    float vibrational_energy,    // UV energy deposited (0 for LIF)
    int n_nearby_excited,        // excited aromatics in range
    float wd_change,             // |water_density - water_density_prev| for SDST energy_gradient
    unsigned int phase_bits      // 10-bit CCNS phase from ProtocolState (REQUIRED — no default)
) {
    event.timestep = timestep;
    event.voxel_idx = voxel_idx;
    event.position = voxel_center;
    event.intensity = intensity;
    event.spike_source = spike_source;
    event.wavelength_nm = wavelength_nm;
    event.aromatic_type = aromatic_type;
    event.aromatic_residue_id = aromatic_residue_id;
    event.water_density = water_density;
    event.vibrational_energy = vibrational_energy;
    event.n_nearby_excited = n_nearby_excited;
    event.wd_change = wd_change;
    event.phase_bits = phase_bits; // 10-bit CCNS phase from ProtocolState
    event.n_residues = 0;

    // Map to nearby residues via warp matrix
    int seen_residues[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

    for (int i = 0; i < warp_entry.n_atoms && event.n_residues < 8; i++) {
        int atom_idx = warp_entry.atom_indices[i];
        int res_id = residue_ids[atom_idx];
        // -1 is the canonical non-analysis sentinel for water, ions and
        // lambda-fragment atoms. It must never materialize as a Ghost/KCC
        // residue identity.
        if (res_id < 0) continue;

        // Check if already seen
        bool seen = false;
        for (int j = 0; j < event.n_residues; j++) {
            if (seen_residues[j] == res_id) {
                seen = true;
                break;
            }
        }

        if (!seen) {
            event.nearby_residues[event.n_residues] = res_id;
            seen_residues[event.n_residues] = res_id;
            event.n_residues++;
        }
    }
}

// ============================================================================
// MAIN FUSED KERNEL - HYPEROPTIMIZED
// ============================================================================
// Optimizations applied:
// 1. __launch_bounds__ for occupancy tuning (256 threads, 4 blocks/SM)
// 2. __restrict__ for pointer aliasing hints
// 3. #pragma unroll for hot loops (applied in body)
// 4. __ldg() for read-only L2 cache hints (applied in body)
// 5. Warp shuffle reductions where applicable

// ── P5 Step 1 — charge-neutral excited-state charge rescale ───────────────
//
// An electronic excitation changes a chromophore's DIPOLE, not its net charge.
// The historical form `q_ground * scale` is a uniform multiplicative rescale,
// which preserves the set's monopole only when that set already sums to zero.
//
// That held BY ACCIDENT for the legacy atom mapping (the first 16 atoms of each
// aromatic residue sum to about +0.003 e). It FAILS for the true ring atom set,
// which sums to about -0.725 e per aromatic. Swapping the mapping to ring atoms
// under the multiplicative form would inject roughly -16 e across a 30-aromatic
// system directly into the Coulomb sum — and every geometry check would still
// pass while it happened.
//
// The affine form rescales about the set's own mean charge:
//     q_new = q_mean + (q_ground - q_mean) * scale
// so sum(q_new) == sum(q_ground) for ANY scale. The dipole is rescaled, the
// monopole is conserved exactly, and the mapping can safely become ring-local.
__device__ __forceinline__ float uv_excited_charge(
    float q_ground,
    const float* d_aromatic_charge_set_mean,
    int aromatic_idx,
    float scale
) {
    if (d_aromatic_charge_set_mean == nullptr) {
        return q_ground * scale;
    }
    const float q_mean = d_aromatic_charge_set_mean[aromatic_idx];
    return q_mean + (q_ground - q_mean) * scale;
}

// Phase specialization changes compiler context, never the phase map.
// Each export retains all109 arguments and checks its runtime phase before
// entering this shared original body. The legacy <-1> route remains complete.
template<bool Compensated>
__device__ __forceinline__ void bp_native_asc_measure_kick(
    const BiprismCoordinateOwnerFfi* owner,uint32_t* fault,
    PrismAscDeviceV1* asc,uint32_t atom,uint32_t atoms,uint32_t phase,
    const float* x,const float* force,const float* velocity,float scale,float limit,
    float mass,float dt,float conversion) {
    if constexpr(Compensated) {
        const auto p=BpNativeGeometry<true>::load(nullptr,(int)atom,owner,fault);
        const double point[3]={p.x,p.y,p.z};
        prism_asc_measure_kick_f64(asc,atom,atoms,phase,point,force,velocity,scale,limit,mass,dt,conversion);
    } else prism_asc_measure_kick(asc,atom,atoms,phase,x,force,velocity,scale,limit,mass,dt,conversion);
}

template<int StaticPhase, bool BiprismOwnership = false, bool Compensated = false>
__device__ __forceinline__ void prism_amber_phase_body_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int runtime_execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership = nullptr,
    const BiprismCoordinateOwnerFfi* coordinate_owner = nullptr,
    uint32_t* precision_fault = nullptr

) {
    if (StaticPhase >= 0 && runtime_execution_phase != StaticPhase) {
        PRISM_PHYSICS_TRAP();
        return;
    }
    const int execution_phase = StaticPhase < 0 ? runtime_execution_phase : StaticPhase;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    using G=BpNativeGeometry<Compensated>;
    using GReal=typename G::Real;
    using GVec=typename G::Vec;
    if constexpr(Compensated) {
        if(!BiprismOwnership || !bp_native_coordinate_owner_valid(coordinate_owner,
                reinterpret_cast<const float*>(positions),n_atoms,
                BP_COORD_CURRENT_INITIALIZED|BP_COORD_START_INITIALIZED,precision_fault,
                reinterpret_cast<const float*>(positions_step_start)))return;
        if(tid<n_atoms) {
            const auto point=G::load(positions,tid,coordinate_owner,precision_fault);
            if(!isfinite(point.x)||!isfinite(point.y)||!isfinite(point.z))return;
        }
    }


    // New-only profile: the historical May negative flag has a different
    // no-418.4 propagation convention and is not admitted by the AU bridge.
    // The false template specialization erases all ownership reads/checks.
    if (BiprismOwnership) {
        if (d_biprism_ownership == nullptr || exact_nonbonded_exceptions_enabled < 0) {
            PRISM_PHYSICS_TRAP();
            return;
        }
        if (tid < n_atoms && !prism_biprism_ownership_valid_v1(d_biprism_ownership[tid])) {
            PRISM_PHYSICS_TRAP();
            return;
        }
    }


    // Reconstruct grid_origin from individual values
    float3 grid_origin = make_float3(grid_origin_x, grid_origin_y, grid_origin_z);

    // ════════════════════════════════════════════════════════════════════════
    // Load dynamic state from ProtocolState into registers (ONCE per thread).
    // The Director kernel wrote these values on the SAME stream before this
    // kernel launched — CUDA stream ordering guarantees visibility.
    //
    // After this block, all local variables are in registers — identical
    // performance to the original scalar parameter path.
    // ════════════════════════════════════════════════════════════════════════
    const float target_temp = d_protocol->current_temperature;
    const int uv_burst_active = d_protocol->uv_burst_active;
    const float uv_wavelength_nm = d_protocol->uv_wavelength_nm;
    const float dt = d_protocol->dt;
    const float gamma = d_protocol->effective_gamma;
    const int timestep = (int)d_protocol->current_step;

    // ========================================================================
    // PHASE 1: AMBER FORCE COMPUTATION
    // ========================================================================

    const bool deterministic_force_gather =
        atom_bond_list != nullptr && atom_bond_count != nullptr && atom_bond_position != nullptr &&
        atom_angle_list != nullptr && atom_angle_count != nullptr && atom_angle_position != nullptr &&
        atom_dihedral_list != nullptr && atom_dihedral_count != nullptr && atom_dihedral_position != nullptr &&
        use_neighbor_list && neighbor_list != nullptr && n_neighbors != nullptr;

    // ========================================================================
    // DETERMINISTIC AMBER FORCE COMPUTATION
    // ========================================================================
    //
    // Each atom owns its force row and gathers all bonded/nonbonded
    // contributions in a fixed local order. This removes f32 force atomics. The
    // execution_phase launch boundary below provides the required grid-wide
    // ordering before Velocity Verlet integration begins.
    float cutoff_sq = cutoff * cutoff;

    if ((execution_phase == MD_PHASE_FORCE_N || execution_phase == MD_PHASE_FORCE_NP1)
        && deterministic_force_gather) {
        if (tid < n_atoms) {
            float3 local_force = make_float3(0.0f, 0.0f, 0.0f);
            double local_pe = 0.0;
            double local_lambda_dudl = 0.0;
            double local_lambda_energy = 0.0;
            const int my_lambda_slot = lambda_atom_slot != nullptr ? lambda_atom_slot[tid] : -1;
            if (my_lambda_slot >= n_lambda_slots) PRISM_PHYSICS_TRAP();

            const int bond_base = tid * max_bonds_per_atom;
            const int nb = atom_bond_count[tid];
            for (int ib = 0; ib < nb; ib++) {
                const int bond_idx = atom_bond_list[bond_base + ib];
                const int pos = atom_bond_position[bond_base + ib];
                BondParam bond = bonds[bond_idx];
                float3 fi = make_float3(0, 0, 0);
                float3 fj = make_float3(0, 0, 0);
                const GVec pi = G::load(positions,bond.i,coordinate_owner,precision_fault);
                const GVec pj = G::load(positions,bond.j,coordinate_owner,precision_fault);
                compute_bond_force(pi, pj, bond.r0, bond.k, fi, fj, nullptr, bond.i, bond.j);
                const float3 mine = (pos == 0) ? fi : fj;
                local_force.x += mine.x;
                local_force.y += mine.y;
                local_force.z += mine.z;

                const GVec rij = G::make(pj.x - pi.x, pj.y - pi.y, pj.z - pi.z);
                const GReal r = bp_native_sqrt(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z);
                if (r >= 1e-8f) {
                    const GReal dr = r - bond.r0;
                    local_pe += (double)bond.k * (double)dr * (double)dr * 0.5;
                }
            }

            const int angle_base = tid * max_angles_per_atom;
            const int na = atom_angle_count[tid];
            for (int ia = 0; ia < na; ia++) {
                const int angle_idx = atom_angle_list[angle_base + ia];
                const int pos = atom_angle_position[angle_base + ia];
                AngleParam angle = angles[angle_idx];
                float3 fi = make_float3(0, 0, 0);
                float3 fj = make_float3(0, 0, 0);
                float3 fk = make_float3(0, 0, 0);
                const GVec pi = G::load(positions,angle.i,coordinate_owner,precision_fault);
                const GVec pj = G::load(positions,angle.j,coordinate_owner,precision_fault);
                const GVec pk = G::load(positions,angle.k,coordinate_owner,precision_fault);
                compute_angle_force(pi, pj, pk, angle.theta0, angle.force_k,
                                    fi, fj, fk, nullptr, angle.i, angle.j, angle.k);
                const float3 mine = (pos == 0) ? fi : ((pos == 1) ? fj : fk);
                local_force.x += mine.x;
                local_force.y += mine.y;
                local_force.z += mine.z;

                const GVec rij = G::make(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                const GVec rkj = G::make(pk.x - pj.x, pk.y - pj.y, pk.z - pj.z);
                const GReal rij_len = bp_native_sqrt(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z);
                const GReal rkj_len = bp_native_sqrt(rkj.x * rkj.x + rkj.y * rkj.y + rkj.z * rkj.z);
                if (rij_len >= 1e-8f && rkj_len >= 1e-8f) {
                    GReal cos_theta = (rij.x * rkj.x + rij.y * rkj.y + rij.z * rkj.z) / (rij_len * rkj_len);
                    cos_theta = bp_native_max(-1.0f, bp_native_min(1.0f, cos_theta));
                    const GReal dtheta = bp_native_acos(cos_theta) - angle.theta0;
                    local_pe += (double)angle.force_k * (double)dtheta * (double)dtheta * (1.0 / 3.0);
                }
            }

            const int dihedral_base = tid * max_dihedrals_per_atom;
            const int nd = atom_dihedral_count[tid];
            for (int idh = 0; idh < nd; idh++) {
                const int dih_idx = atom_dihedral_list[dihedral_base + idh];
                const int pos = atom_dihedral_position[dihedral_base + idh];
                DihedralParam dih = dihedrals[dih_idx];
                float3 fi = make_float3(0, 0, 0);
                float3 fj = make_float3(0, 0, 0);
                float3 fk = make_float3(0, 0, 0);
                float3 fl = make_float3(0, 0, 0);
                const GVec pi = G::load(positions,dih.i,coordinate_owner,precision_fault);
                const GVec pj = G::load(positions,dih.j,coordinate_owner,precision_fault);
                const GVec pk = G::load(positions,dih.k,coordinate_owner,precision_fault);
                const GVec pl = G::load(positions,dih.l,coordinate_owner,precision_fault);
                compute_dihedral_force(pi, pj, pk, pl, dih.periodicity, dih.phase, dih.force_k,
                                       fi, fj, fk, fl, nullptr, dih.i, dih.j, dih.k, dih.l);
                const float3 mine = (pos == 0) ? fi : ((pos == 1) ? fj : ((pos == 2) ? fk : fl));
                local_force.x += mine.x;
                local_force.y += mine.y;
                local_force.z += mine.z;

                const GVec b1 = G::make(pj.x - pi.x, pj.y - pi.y, pj.z - pi.z);
                const GVec b2 = G::make(pk.x - pj.x, pk.y - pj.y, pk.z - pj.z);
                const GVec b3 = G::make(pl.x - pk.x, pl.y - pk.y, pl.z - pk.z);
                const GVec c1 = G::make(
                    b1.y * b2.z - b1.z * b2.y,
                    b1.z * b2.x - b1.x * b2.z,
                    b1.x * b2.y - b1.y * b2.x);
                const GVec c2 = G::make(
                    b2.y * b3.z - b2.z * b3.y,
                    b2.z * b3.x - b2.x * b3.z,
                    b2.x * b3.y - b2.y * b3.x);
                const GReal c1_len = bp_native_sqrt(c1.x * c1.x + c1.y * c1.y + c1.z * c1.z);
                const GReal c2_len = bp_native_sqrt(c2.x * c2.x + c2.y * c2.y + c2.z * c2.z);
                if (c1_len >= 1e-8f && c2_len >= 1e-8f) {
                    GReal cos_phi = (c1.x * c2.x + c1.y * c2.y + c1.z * c2.z) / (c1_len * c2_len);
                    cos_phi = bp_native_max(-1.0f, bp_native_min(1.0f, cos_phi));
                    const GReal sign = c1.x * b3.x + c1.y * b3.y + c1.z * b3.z;
                    GReal phi = bp_native_acos(cos_phi);
                    if (sign < 0.0f) phi = -phi;
                    local_pe += (double)dih.force_k *
                                (1.0 + cos((double)dih.periodicity * (double)phi - (double)dih.phase)) *
                                0.25;
                }
            }

            float my_charge = charges[tid];
            float my_charge_scale = 1.0f;
            const int my_arom = d_atom_to_aromatic[tid];
            if (my_arom >= 0 && d_is_excited[my_arom]) {
                const float pop = d_electronic_population[my_arom];
                float ratio_sqrt;
                switch (d_aromatic_type[my_arom]) {
                    case 0: ratio_sqrt = TRP_DIPOLE_RATIO_SQRT; break;
                    case 1: ratio_sqrt = TYR_DIPOLE_RATIO_SQRT; break;
                    case 2: ratio_sqrt = PHE_DIPOLE_RATIO_SQRT; break;
                    default: ratio_sqrt = 1.0f;
                }
                const float scale = 1.0f + (ratio_sqrt - 1.0f) * pop;
                my_charge_scale = scale;
                my_charge = uv_excited_charge(
                    d_ground_state_charges[tid], exact_nonbonded_exceptions_enabled < 0 ? nullptr : d_aromatic_charge_set_mean, my_arom, scale);
            }
            if (lambda_v2_ewald_enabled != 0) {
                // The production lambda charge plane is the sole charge
                // authority for both real-space Ewald and reciprocal PME.
                // UV excited-state rescaling is not composable with this
                // Hamiltonian unless it is represented as an explicit lambda
                // endpoint and is therefore deliberately bypassed here.
                my_charge = lambda_current_charges[tid];
                my_charge_scale = 1.0f;
            }

            const GVec my_pos = G::load(positions,tid,coordinate_owner,precision_fault);
            const LJParam my_lj = lj_params[tid];
            const int my_n_neighbors = __ldg(&n_neighbors[tid]);
            const int* my_neighbors = &neighbor_list[tid * NEIGHBOR_LIST_SIZE];
            for (int k = 0; k < my_n_neighbors; k++) {
                const int j = __ldg(&my_neighbors[k]);
                int pair14_index = -1;
                const int s14 = pairs14_offsets[tid];
                const int e14 = pairs14_offsets[tid + 1];
                for (int p = s14; p < e14; p++) {
                    if (pairs14_list[p] == j) { pair14_index = p; break; }
                }
                // Exact OpenMM exceptions are evaluated in a separate CSR
                // loop below so they are independent of neighbor cutoff.
                if (exact_nonbonded_exceptions_enabled > 0 && pair14_index >= 0) continue;
                const GVec other_pos = G::load(positions,j,coordinate_owner,precision_fault);
                GVec displacement = G::make(
                    my_pos.x - other_pos.x,
                    my_pos.y - other_pos.y,
                    my_pos.z - other_pos.z);
                if (lambda_v2_ewald_enabled != 0) {
                    displacement = minimum_image_orthorhombic_v1(
                        displacement,
                        make_float3(lambda_box_x, lambda_box_y, lambda_box_z));
                }
                const GReal r2 = displacement.x * displacement.x
                               + displacement.y * displacement.y
                               + displacement.z * displacement.z;
                if (r2 >= cutoff_sq || r2 < 0.01f) continue;

                float other_charge = __ldg(&charges[j]);
                float other_charge_scale = 1.0f;
                const int other_arom = d_atom_to_aromatic[j];
                if (other_arom >= 0 && d_is_excited[other_arom]) {
                    const float pop = d_electronic_population[other_arom];
                    float ratio_sqrt;
                    switch (d_aromatic_type[other_arom]) {
                        case 0: ratio_sqrt = TRP_DIPOLE_RATIO_SQRT; break;
                        case 1: ratio_sqrt = TYR_DIPOLE_RATIO_SQRT; break;
                        case 2: ratio_sqrt = PHE_DIPOLE_RATIO_SQRT; break;
                        default: ratio_sqrt = 1.0f;
                    }
                    const float scale = 1.0f + (ratio_sqrt - 1.0f) * pop;
                    other_charge_scale = scale;
                    other_charge = uv_excited_charge(
                        d_ground_state_charges[j], exact_nonbonded_exceptions_enabled < 0 ? nullptr : d_aromatic_charge_set_mean, other_arom, scale);
                }
                if (lambda_v2_ewald_enabled != 0) {
                    other_charge = lambda_current_charges[j];
                    other_charge_scale = 1.0f;
                }

                const LJParam other_lj = lj_params[j];
                const bool is_14 = pair14_index >= 0;
                const bool exact_14 = is_14 && exact_nonbonded_exceptions_enabled > 0;
                // The prmtop exception stores the ground-state charge product.
                // Preserve that exact pair identity while applying the same UV
                // excited-state charge multipliers as ordinary nonbonded pairs.
                const float exact_charge_product = exact_14
                    ? pairs14_charge_product[pair14_index] * my_charge_scale * other_charge_scale
                    : 0.0f;
                const float exact_sigma = exact_14 ? pairs14_sigma[pair14_index] : 0.0f;
                const float exact_epsilon = exact_14 ? pairs14_epsilon[pair14_index] : 0.0f;

                float3 fi = make_float3(0, 0, 0);
                float3 fj = make_float3(0, 0, 0);
                const int other_lambda_slot = lambda_atom_slot != nullptr ? lambda_atom_slot[j] : -1;
                if (other_lambda_slot >= n_lambda_slots) PRISM_PHYSICS_TRAP();
                const bool lambda_pair = my_lambda_slot >= 0 || other_lambda_slot >= 0;
                double lambda_pair_pe = 0.0;
                double lambda_pair_dudl = 0.0;
                if (lambda_v2_ewald_enabled != 0) {
                    const float li = my_lambda_slot >= 0 ? lambda_slot_values[my_lambda_slot] : 1.0f;
                    const float lj = other_lambda_slot >= 0 ? lambda_slot_values[other_lambda_slot] : 1.0f;
                    const bool same_slot_internal = my_lambda_slot >= 0
                        && other_lambda_slot == my_lambda_slot;
                    double lj_pe = 0.0;
                    double lj_dudl_pair = 0.0;
                    const bool ordinary_owns_lj = !lambda_pair;
                    if (ordinary_owns_lj) {
                        compute_lambda_softcore_lj_v2_pair(
                            displacement, G::make(0.0f, 0.0f, 0.0f),
                            exact_14 ? exact_sigma : 0.5f * (my_lj.sigma + other_lj.sigma),
                            exact_14 ? exact_epsilon : bp_native_sqrt(my_lj.epsilon * other_lj.epsilon),
                            lambda_pair ? li * lj : 1.0f,
                            lambda_alpha_lj, lambda_softcore_power, lambda_coupling_power,
                            cutoff_sq, exact_14, fi, lj_pe, lj_dudl_pair);
                    }
                    double ewald_pe = 0.0;
                    double ewald_dudl = 0.0;
                    if (same_slot_internal) {
                        double desired_q_product = 0.0;
                        double desired_d_q_product = 0.0;
                        lambda_desired_internal_charge_product_v1(
                            tid, j, my_lambda_slot, li,
                            lambda_endpoint_a_charges,
                            lambda_endpoint_b_charges,
                            lambda_endpoint_occupancy_masks,
                            desired_q_product, desired_d_q_product);
                        // The neighbor list is atom-owned and evaluates the
                        // same intramolecular pair once from each endpoint.
                        // Energy is split below; split the full endpoint-plane
                        // derivative here as well so the slot reduction sees
                        // d[(1-lambda)UA+lambda UB]/dlambda exactly once.
                        compute_charge_product_ewald_realspace_v2(
                            displacement, desired_q_product,
                            0.5 * desired_d_q_product,
                            lambda_pme_beta_a_inv,
                            fi, ewald_pe, ewald_dudl);
                    } else {
                        const float my_partial_dq = my_lambda_slot >= 0
                            ? lambda_charge_derivatives[tid] : 0.0f;
                        compute_dynamic_charge_ewald_realspace_v2(
                            displacement, my_charge, my_partial_dq,
                            other_charge, 0.0f, lambda_pme_beta_a_inv,
                            fi, ewald_pe, ewald_dudl);
                    }
                    lambda_pair_pe = lj_pe + ewald_pe;
                    // Each fragment atom owns the derivative for its own slot.
                    // Receptor-owned mirror rows intentionally do not contribute.
                    if (my_lambda_slot >= 0) {
                        lambda_pair_dudl = lj_dudl_pair * (double)lj + ewald_dudl;
                        local_lambda_dudl += lambda_pair_dudl;
                        // A receptor/fragment interaction belongs wholly to
                        // the fragment slot.  Cross-fragment interactions are
                        // split evenly so summing slot energies is exact.
                        local_lambda_energy += lambda_pair_pe
                            * (other_lambda_slot >= 0 ? 0.5 : 1.0);
                    }
                } else if (lambda_pair) {
                    // Legacy V1 is retained only for explicit non-production
                    // compatibility lanes and is never mixed with PME.
                    const float li = my_lambda_slot >= 0 ? lambda_slot_values[my_lambda_slot] : 1.0f;
                    const float lj = other_lambda_slot >= 0 ? lambda_slot_values[other_lambda_slot] : 1.0f;
                    compute_lambda_softcore_pair(
                        my_pos, other_pos,
                        exact_14 ? exact_charge_product : my_charge * other_charge,
                        exact_14 ? exact_sigma : 0.5f * (my_lj.sigma + other_lj.sigma),
                        exact_14 ? exact_epsilon : bp_native_sqrt(my_lj.epsilon * other_lj.epsilon),
                        li * lj,
                        lambda_alpha_lj, lambda_alpha_q_a2,
                        lambda_softcore_power, lambda_coupling_power,
                        cutoff_sq, exact_14, fi, lambda_pair_pe, lambda_pair_dudl);
                    if (my_lambda_slot >= 0) {
                        local_lambda_dudl += lambda_pair_dudl * (double)lj;
                        local_lambda_energy += lambda_pair_pe
                            * (other_lambda_slot >= 0 ? 0.5 : 1.0);
                    }
                } else {
                    compute_nonbonded_force(
                        my_pos, other_pos,
                        my_charge, other_charge,
                        my_lj.sigma, my_lj.epsilon,
                        other_lj.sigma, other_lj.epsilon,
                        fi, fj, cutoff_sq,
                        nullptr, tid, j,
                        is_14, exact_14, exact_charge_product, exact_sigma, exact_epsilon, exact_nonbonded_exceptions_enabled < 0
                    );
                }
                local_force.x += fi.x;
                local_force.y += fi.y;
                local_force.z += fi.z;

                const GReal inv_r = bp_native_rsqrt(r2);
                const float sigma = exact_14 ? exact_sigma : 0.5f * (my_lj.sigma + other_lj.sigma);
                const float epsilon = exact_14 ? exact_epsilon : bp_native_sqrt(my_lj.epsilon * other_lj.epsilon);
                const float charge_product = exact_14 ? exact_charge_product : my_charge * other_charge;
                const GReal sigma_r = sigma * inv_r;
                const GReal sigma_r2 = sigma_r * sigma_r;
                const GReal sigma_r6 = sigma_r2 * sigma_r2 * sigma_r2;
                const GReal sigma_r12 = sigma_r6 * sigma_r6;
                const float scale_lj = (is_14 && !exact_14) ? FUDGE_LJ : 1.0f;
                const float scale_qq = (is_14 && !exact_14) ? FUDGE_QQ : 1.0f;
                const double v_lj = 4.0 * (double)epsilon * ((double)sigma_r12 - (double)sigma_r6);
                const double v_elec = (double)COULOMB_CONSTANT * (double)charge_product *
                                      (double)inv_r;
                local_pe += lambda_v2_ewald_enabled != 0
                    ? lambda_pair_pe * 0.5
                    : (lambda_pair
                        ? lambda_pair_pe
                        : (scale_lj * v_lj + scale_qq * v_elec
                            - (exact_nonbonded_exceptions_enabled < 0 ? 0.0 : ordinary_cutoff_potential_offset_v1<GReal>(sigma,epsilon,
                                charge_product,cutoff_sq,scale_lj,scale_qq)))) * 0.5;
            }

            if (exact_nonbonded_exceptions_enabled > 0) {
                const int s14 = pairs14_offsets[tid];
                const int e14 = pairs14_offsets[tid + 1];
                for (int p = s14; p < e14; p++) {
                    const int j = pairs14_list[p];
                    const GVec other_pos = G::load(positions,j,coordinate_owner,precision_fault);
                    GVec displacement = G::make(
                        my_pos.x - other_pos.x,
                        my_pos.y - other_pos.y,
                        my_pos.z - other_pos.z);
                    if (lambda_v2_ewald_enabled != 0) {
                        displacement = minimum_image_orthorhombic_v1(
                            displacement,
                            make_float3(lambda_box_x, lambda_box_y, lambda_box_z));
                    }
                    const GReal r2 = displacement.x * displacement.x
                                   + displacement.y * displacement.y
                                   + displacement.z * displacement.z;
                    if (r2 < 1e-8f) {
                        // Omitting an explicit OpenMM exception would corrupt
                        // the force field; coincident exact-pair atoms are a
                        // fatal topology/runtime state.
                        PRISM_PHYSICS_TRAP();
                    }

                    float other_charge_scale = 1.0f;
                    const int other_arom = d_atom_to_aromatic[j];
                    if (other_arom >= 0 && d_is_excited[other_arom]) {
                        const float pop = d_electronic_population[other_arom];
                        float ratio_sqrt;
                        switch (d_aromatic_type[other_arom]) {
                            case 0: ratio_sqrt = TRP_DIPOLE_RATIO_SQRT; break;
                            case 1: ratio_sqrt = TYR_DIPOLE_RATIO_SQRT; break;
                            case 2: ratio_sqrt = PHE_DIPOLE_RATIO_SQRT; break;
                            default: ratio_sqrt = 1.0f;
                        }
                        // P5 Step 1: this path scales a PRECOMPUTED
                        // pairs14_charge_product (an OpenMM exception product,
                        // not a per-atom charge), so it contributes no monopole
                        // and needs no charge-neutral correction. Recomputing it
                        // from individual charges would change the 1-4 exception
                        // contract. Multiplicative BY DESIGN, not by omission.
                        other_charge_scale = 1.0f + (ratio_sqrt - 1.0f) * pop;
                    }
                    const float other_charge = lambda_v2_ewald_enabled != 0
                        ? lambda_current_charges[j] : charges[j];
                    if (lambda_v2_ewald_enabled != 0) other_charge_scale = 1.0f;

                    const float exact_charge_product =
                        pairs14_charge_product[p] * my_charge_scale * other_charge_scale;
                    const float exact_sigma = pairs14_sigma[p];
                    const float exact_epsilon = pairs14_epsilon[p];
                    float3 fi = make_float3(0, 0, 0);
                    float3 fj = make_float3(0, 0, 0);
                    const LJParam other_lj = lj_params[j];
                    const int other_lambda_slot = lambda_atom_slot != nullptr ? lambda_atom_slot[j] : -1;
                    if (other_lambda_slot >= n_lambda_slots) PRISM_PHYSICS_TRAP();
                    const bool lambda_pair = my_lambda_slot >= 0 || other_lambda_slot >= 0;
                    double lambda_pair_pe = 0.0;
                    double lambda_pair_dudl = 0.0;
                    if (lambda_v2_ewald_enabled != 0) {
                        const float li = my_lambda_slot >= 0 ? lambda_slot_values[my_lambda_slot] : 1.0f;
                        const float lj = other_lambda_slot >= 0 ? lambda_slot_values[other_lambda_slot] : 1.0f;
                        const bool same_slot_internal = my_lambda_slot >= 0
                            && other_lambda_slot == my_lambda_slot;
                        double lj_pe = 0.0;
                        double lj_dudl_pair = 0.0;
                        const bool ordinary_owns_lj = !lambda_pair;
                        if (ordinary_owns_lj) {
                            compute_lambda_softcore_lj_v2_pair(
                                displacement, G::make(0.0f, 0.0f, 0.0f),
                                exact_sigma, exact_epsilon,
                                lambda_pair ? li * lj : 1.0f,
                                lambda_alpha_lj, lambda_softcore_power, lambda_coupling_power,
                                cutoff_sq, true, fi, lj_pe, lj_dudl_pair);
                        }
                        float regular_q_product = my_charge * other_charge;
                        float d_regular_q_product = my_lambda_slot >= 0
                            ? lambda_charge_derivatives[tid] * other_charge : 0.0f;
                        if (same_slot_internal) {
                            double desired_q_product = 0.0;
                            double desired_d_q_product = 0.0;
                            lambda_desired_internal_charge_product_v1(
                                tid, j, my_lambda_slot, li,
                                lambda_endpoint_a_charges,
                                lambda_endpoint_b_charges,
                                lambda_endpoint_occupancy_masks,
                                desired_q_product, desired_d_q_product);
                            regular_q_product = (float)desired_q_product;
                            d_regular_q_product = (float)(0.5 * desired_d_q_product);
                        }
                        const float exact_q_product = lambda_pair
                            ? exact_charge_product * li * lj : exact_charge_product;
                        const float d_exact_q_product = my_lambda_slot >= 0
                            ? exact_charge_product * lj : 0.0f;
                        double exception_pe = 0.0;
                        double exception_dudl = 0.0;
                        compute_dynamic_charge_ewald_exception_v2(
                            displacement,
                            regular_q_product, d_regular_q_product,
                            exact_q_product, d_exact_q_product,
                            lambda_pme_beta_a_inv,
                            fi, exception_pe, exception_dudl);
                        lambda_pair_pe = lj_pe + exception_pe;
                        if (my_lambda_slot >= 0) {
                            lambda_pair_dudl = lj_dudl_pair * (double)lj + exception_dudl;
                            local_lambda_dudl += lambda_pair_dudl;
                            local_lambda_energy += lambda_pair_pe
                                * (other_lambda_slot >= 0 ? 0.5 : 1.0);
                        }
                    } else if (lambda_pair) {
                        const float li = my_lambda_slot >= 0 ? lambda_slot_values[my_lambda_slot] : 1.0f;
                        const float lj = other_lambda_slot >= 0 ? lambda_slot_values[other_lambda_slot] : 1.0f;
                        compute_lambda_softcore_pair(
                            my_pos, other_pos, exact_charge_product, exact_sigma, exact_epsilon,
                            li * lj, lambda_alpha_lj, lambda_alpha_q_a2,
                            lambda_softcore_power, lambda_coupling_power,
                            cutoff_sq, true, fi, lambda_pair_pe, lambda_pair_dudl);
                        if (my_lambda_slot >= 0) {
                            local_lambda_dudl += lambda_pair_dudl * (double)lj;
                            local_lambda_energy += lambda_pair_pe
                                * (other_lambda_slot >= 0 ? 0.5 : 1.0);
                        }
                    } else {
                        compute_nonbonded_force(
                            my_pos, other_pos,
                            my_charge, other_charge,
                            my_lj.sigma, my_lj.epsilon,
                            other_lj.sigma, other_lj.epsilon,
                            fi, fj, cutoff_sq,
                            nullptr, tid, j,
                            true, true, exact_charge_product, exact_sigma, exact_epsilon, exact_nonbonded_exceptions_enabled < 0
                        );
                    }
                    local_force.x += fi.x;
                    local_force.y += fi.y;
                    local_force.z += fi.z;

                    const GReal inv_r = bp_native_rsqrt(r2);
                    const GReal sigma_r = exact_sigma * inv_r;
                    const GReal sigma_r2 = sigma_r * sigma_r;
                    const GReal sigma_r6 = sigma_r2 * sigma_r2 * sigma_r2;
                    const GReal sigma_r12 = sigma_r6 * sigma_r6;
                    const double v_lj = 4.0 * (double)exact_epsilon
                        * ((double)sigma_r12 - (double)sigma_r6);
                    const double v_elec = (double)COULOMB_CONSTANT
                        * (double)exact_charge_product * (double)inv_r;
                    local_pe += lambda_v2_ewald_enabled != 0
                        ? lambda_pair_pe * 0.5
                        : (lambda_pair ? lambda_pair_pe : (v_lj + v_elec)) * 0.5;
                }
            }

            if constexpr(Compensated) {
                if(atomicAdd(precision_fault,0u)!=0u || !isfinite(local_force.x)||!isfinite(local_force.y)||
                   !isfinite(local_force.z)||!isfinite(local_pe)||!isfinite(local_lambda_dudl)||
                   !isfinite(local_lambda_energy)) {bp_native_coordinate_fault(precision_fault);return;}
            }
            forces[tid] = local_force;
            if (pe_components != nullptr) {
                pe_components[tid] = local_pe;
            }
            if (lambda_dudl_atom != nullptr) {
                lambda_dudl_atom[tid] = local_lambda_dudl;
            }
            if (lambda_energy_atom != nullptr) {
                lambda_energy_atom[tid] = local_lambda_energy;
            }
        }
    } else if (execution_phase == MD_PHASE_FORCE_N || execution_phase == MD_PHASE_FORCE_NP1) {
        // Legacy atomic force path retained as a hard fallback for non-production
        // harnesses that have not uploaded incidence lists.
        //
        // Production teacher generation must never reach this path: f32
        // force atomics are order-dependent and were shown to leak into
        // residue-level spike supervision. Trap instead of silently emitting
        // nondeterministic trajectories if host-side incidence setup regresses.
        PRISM_PHYSICS_TRAP();
        return;

        if (tid < n_atoms) {
            forces[tid] = make_float3(0.0f, 0.0f, 0.0f);
        }
        __syncthreads();

        #pragma unroll 2
        for (int b = tid; b < n_bonds; b += gridDim.x * blockDim.x) {
            BondParam bond = bonds[b];
            int bi = bond.i, bj = bond.j;
            float3 fi = make_float3(0, 0, 0);
            float3 fj = make_float3(0, 0, 0);
            compute_bond_force(positions[bi], positions[bj], bond.r0, bond.k, fi, fj,
                               pe_components, bi, bj);
            atomicAdd(&forces[bi].x, fi.x);
            atomicAdd(&forces[bi].y, fi.y);
            atomicAdd(&forces[bi].z, fi.z);
            atomicAdd(&forces[bj].x, fj.x);
            atomicAdd(&forces[bj].y, fj.y);
            atomicAdd(&forces[bj].z, fj.z);
        }

        #pragma unroll 2
        for (int a = tid; a < n_angles; a += gridDim.x * blockDim.x) {
            AngleParam angle = angles[a];
            float3 fi = make_float3(0, 0, 0);
            float3 fj = make_float3(0, 0, 0);
            float3 fk = make_float3(0, 0, 0);
            compute_angle_force(positions[angle.i], positions[angle.j], positions[angle.k],
                                angle.theta0, angle.force_k, fi, fj, fk,
                                pe_components, angle.i, angle.j, angle.k);
            atomicAdd(&forces[angle.i].x, fi.x);
            atomicAdd(&forces[angle.i].y, fi.y);
            atomicAdd(&forces[angle.i].z, fi.z);
            atomicAdd(&forces[angle.j].x, fj.x);
            atomicAdd(&forces[angle.j].y, fj.y);
            atomicAdd(&forces[angle.j].z, fj.z);
            atomicAdd(&forces[angle.k].x, fk.x);
            atomicAdd(&forces[angle.k].y, fk.y);
            atomicAdd(&forces[angle.k].z, fk.z);
        }

        #pragma unroll 2
        for (int d = tid; d < n_dihedrals; d += gridDim.x * blockDim.x) {
            DihedralParam dih = dihedrals[d];
            float3 fi = make_float3(0, 0, 0);
            float3 fj = make_float3(0, 0, 0);
            float3 fk = make_float3(0, 0, 0);
            float3 fl = make_float3(0, 0, 0);
            compute_dihedral_force(positions[dih.i], positions[dih.j], positions[dih.k], positions[dih.l],
                                   dih.periodicity, dih.phase, dih.force_k, fi, fj, fk, fl,
                                   pe_components, dih.i, dih.j, dih.k, dih.l);
            atomicAdd(&forces[dih.i].x, fi.x);
            atomicAdd(&forces[dih.i].y, fi.y);
            atomicAdd(&forces[dih.i].z, fi.z);
            atomicAdd(&forces[dih.j].x, fj.x);
            atomicAdd(&forces[dih.j].y, fj.y);
            atomicAdd(&forces[dih.j].z, fj.z);
            atomicAdd(&forces[dih.k].x, fk.x);
            atomicAdd(&forces[dih.k].y, fk.y);
            atomicAdd(&forces[dih.k].z, fk.z);
            atomicAdd(&forces[dih.l].x, fl.x);
            atomicAdd(&forces[dih.l].y, fl.y);
            atomicAdd(&forces[dih.l].z, fl.z);
        }

        __syncthreads();

        if (tid < n_atoms) {
            float3 my_pos = positions[tid];
            float my_charge = charges[tid];
            LJParam my_lj = lj_params[tid];
            float3 my_force = make_float3(0, 0, 0);

            if (use_neighbor_list && neighbor_list != nullptr && n_neighbors != nullptr) {
                int my_n_neighbors = __ldg(&n_neighbors[tid]);
                const int* my_neighbors = &neighbor_list[tid * NEIGHBOR_LIST_SIZE];
                #pragma unroll 4
                for (int k = 0; k < my_n_neighbors; k++) {
                    int j = __ldg(&my_neighbors[k]);
                    float3 other_pos = positions[j];
                    float dx = my_pos.x - other_pos.x;
                    float dy = my_pos.y - other_pos.y;
                    float dz = my_pos.z - other_pos.z;
                    float r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= cutoff_sq || r2 < 0.01f) continue;

                    float other_charge = __ldg(&charges[j]);
                    LJParam other_lj = lj_params[j];
                    int pair14_index = -1;
                    const int s14 = pairs14_offsets[tid];
                    const int e14 = pairs14_offsets[tid + 1];
                    for (int p = s14; p < e14; p++) {
                        if (pairs14_list[p] == j) { pair14_index = p; break; }
                    }
                    const bool is_14 = pair14_index >= 0;
                    const bool exact_14 = is_14 && exact_nonbonded_exceptions_enabled > 0;
                    const float exact_charge_product = exact_14 ? pairs14_charge_product[pair14_index] : 0.0f;
                    const float exact_sigma = exact_14 ? pairs14_sigma[pair14_index] : 0.0f;
                    const float exact_epsilon = exact_14 ? pairs14_epsilon[pair14_index] : 0.0f;

                    float3 fi = make_float3(0, 0, 0);
                    float3 fj = make_float3(0, 0, 0);
                    compute_nonbonded_force(my_pos, other_pos, my_charge, other_charge,
                                            my_lj.sigma, my_lj.epsilon,
                                            other_lj.sigma, other_lj.epsilon,
                                            fi, fj, cutoff_sq, pe_components, tid, j,
                                            is_14, exact_14, exact_charge_product,
                                            exact_sigma, exact_epsilon, exact_nonbonded_exceptions_enabled < 0);
                    my_force.x += fi.x;
                    my_force.y += fi.y;
                    my_force.z += fi.z;
                    atomicAdd(&forces[j].x, fj.x);
                    atomicAdd(&forces[j].y, fj.y);
                    atomicAdd(&forces[j].z, fj.z);
                }
            } else {
                PRISM_PHYSICS_TRAP();
            }

            atomicAdd(&forces[tid].x, my_force.x);
            atomicAdd(&forces[tid].y, my_force.y);
            atomicAdd(&forces[tid].z, my_force.z);
        }
        __syncthreads();
    }

    if (execution_phase == MD_PHASE_FORCE_N || execution_phase == MD_PHASE_FORCE_NP1) return;
    if (execution_phase < MD_PHASE_FORCE_N
        || (execution_phase > MD_PHASE_KICK_NP1_OU_HALF && execution_phase < MD_PHASE_OU_HALF_N)
        || execution_phase > MD_PHASE_OU_HALF_NP1) {
        PRISM_PHYSICS_TRAP();
        return;
    }


    // Full force evaluation above retains every atom. Only the explicitly
    // selected motion phases below are suppressed for externally owned atoms.
    // Capture the common x_n before core scatter; the graph owner must place
    // this drift/snapshot phase before the accepted v10 state publication.
    if (BiprismOwnership && tid < n_atoms &&
        prism_biprism_external_motion_v1(d_biprism_ownership[tid])) {
        if (execution_phase == MD_PHASE_DRIFT_SAVE_XN) {
            if constexpr(Compensated) {
                if(!G::snapshot(positions_step_start,positions,tid,coordinate_owner,precision_fault))return;
            } else positions_step_start[tid] = positions[tid];
            return;
        }
        if (execution_phase == MD_PHASE_OU_HALF_N || execution_phase == MD_PHASE_OU_HALF_NP1 ||
            execution_phase == MD_PHASE_KICK_N || execution_phase == MD_PHASE_KICK_NP1 ||
            execution_phase == MD_PHASE_OU_HALF_KICK_N || execution_phase == MD_PHASE_KICK_NP1_OU_HALF) {
            return; // No velocity/work/PE writes and no curand state access.
        }
    }

    // Ordinary constrained OVRVO: the bath and kick occupy distinct launches.
    // Project the randomization before applying force, and project the final
    // kick before randomization. This prevents their normal-velocity cross
    // term from being mixed into a later constraint heat receipt. Both maps
    // retain their original exact kinetic-energy work accounting; no shadow
    // integration residual is added to the work ledger.
    if (execution_phase == MD_PHASE_OU_HALF_N || execution_phase == MD_PHASE_OU_HALF_NP1) {
        if (tid < n_atoms && gamma != 0.0f) {
            double* work_component = d_external_work != nullptr ? &d_external_work[tid] : nullptr;
            curandState local_rng = rng_states[tid];
            langevin_thermostat(
                velocities[tid], masses[tid], target_temp, gamma, 0.5f * dt,
                &local_rng, work_component);
            rng_states[tid] = local_rng;
        }
        return;
    }
    if (execution_phase == MD_PHASE_KICK_N || execution_phase == MD_PHASE_KICK_NP1) {
        if (exact_nonbonded_exceptions_enabled < 0) {
            // Post-UV original second half: same unrecomputed, unclamped force.
            // This opcode is admitted only by the historical post-step caller.
            if(execution_phase!=MD_PHASE_KICK_NP1){PRISM_PHYSICS_TRAP();return;}
            if(tid<n_atoms) {
                const float inverse=1.0f/masses[tid];
                velocities[tid].x+=0.5f*dt*forces[tid].x*inverse;
                velocities[tid].y+=0.5f*dt*forces[tid].y*inverse;
                velocities[tid].z+=0.5f*dt*forces[tid].z*inverse;
                if(!isfinite(velocities[tid].x)||!isfinite(velocities[tid].y)||!isfinite(velocities[tid].z)){
                    PRISM_PHYSICS_TRAP();return;
                }
            }
            return;
        }
        if (tid < n_atoms) {
            const int kick_phase = execution_phase == MD_PHASE_KICK_N ? 0 : 1;
            if (gamma == 0.0f) {
                bp_native_asc_measure_kick<Compensated>(coordinate_owner,precision_fault,d_asc_decision,tid,n_atoms,kick_phase,
                    reinterpret_cast<const float*>(&positions[tid]),reinterpret_cast<const float*>(&forces[tid]),
                    reinterpret_cast<const float*>(&velocities[tid]),1.0f,INFINITY,masses[tid],dt,AMBER_KCAL_TO_AMU_A2_PS2);
                apply_exact_force_half_kick(velocities[tid], forces[tid], masses[tid], dt);
            } else {
                double* work_component = d_external_work != nullptr ? &d_external_work[tid] : nullptr;
                bp_native_asc_measure_kick<Compensated>(coordinate_owner,precision_fault,d_asc_decision,tid,n_atoms,kick_phase,
                    reinterpret_cast<const float*>(&positions[tid]),reinterpret_cast<const float*>(&forces[tid]),
                    reinterpret_cast<const float*>(&velocities[tid]),atom_lambda[tid],MAX_INTEGRATOR_FORCE_KCAL_MOL_A,masses[tid],dt,AMBER_KCAL_TO_AMU_A2_PS2);
                apply_force_half_kick_with_intervention_work(
                    velocities[tid], forces[tid], atom_lambda[tid], masses[tid], dt,
                    work_component);
            }
        }
        return;
    }

    // ========================================================================
    // PHASE 1: FIRST OU HALF-STEP + FIRST FORCE HALF-KICK
    // ========================================================================

    if (execution_phase == MD_PHASE_OU_HALF_KICK_N) {
        if(exact_nonbonded_exceptions_enabled < 0) {
            if(tid<n_atoms) {
                curandState local_rng=rng_states[tid];
                if(!prism_may_original_first_kick_bath_drift_v1(
                    positions[tid],positions_step_start[tid],velocities[tid],forces[tid],masses[tid],
                    atom_lambda[tid],target_temp,gamma,dt,&local_rng,
                    pe_components?&pe_components[tid]:nullptr,
                    d_external_work?&d_external_work[tid]:nullptr)) {PRISM_PHYSICS_TRAP();return;}
                rng_states[tid]=local_rng;
            }
            return;
        }
        if (tid < n_atoms) {
            double* work_component = d_external_work != nullptr ? &d_external_work[tid] : nullptr;
            // Dedicated reversible NCMC capture sets gamma to exactly zero.
            // Do not even load/store curand state in that branch: calling the
            // thermostat with c2=0 would leave positions/velocities unchanged
            // but would still consume random numbers and break exact reverse
            // replay plus counter-lineage closure.
            if (gamma == 0.0f) {
                bp_native_asc_measure_kick<Compensated>(coordinate_owner,precision_fault,d_asc_decision,tid,n_atoms,0,
                    reinterpret_cast<const float*>(&positions[tid]),reinterpret_cast<const float*>(&forces[tid]),
                    reinterpret_cast<const float*>(&velocities[tid]),1.0f,INFINITY,masses[tid],dt,AMBER_KCAL_TO_AMU_A2_PS2);
                apply_exact_force_half_kick(
                    velocities[tid], forces[tid], masses[tid], dt);
                return;
            }
            curandState local_rng = rng_states[tid];
            langevin_thermostat(
                velocities[tid], masses[tid], target_temp, gamma, 0.5f * dt,
                &local_rng, work_component);
            bp_native_asc_measure_kick<Compensated>(coordinate_owner,precision_fault,d_asc_decision,tid,n_atoms,0,
                reinterpret_cast<const float*>(&positions[tid]),reinterpret_cast<const float*>(&forces[tid]),
                reinterpret_cast<const float*>(&velocities[tid]),atom_lambda[tid],MAX_INTEGRATOR_FORCE_KCAL_MOL_A,masses[tid],dt,AMBER_KCAL_TO_AMU_A2_PS2);
            apply_force_half_kick_with_intervention_work(
                velocities[tid], forces[tid], atom_lambda[tid], masses[tid], dt,
                work_component);
            rng_states[tid] = local_rng;
        }
        return;
    }

    // Remove any pre-drift bond-parallel velocity before advancing positions.
    if (execution_phase == MD_PHASE_RATTLE_PRE_DRIFT
        || execution_phase == MD_PHASE_RATTLE_AFTER_OU_N
        || execution_phase == MD_PHASE_RATTLE_BEFORE_OU_NP1) {
        for (int c = tid; c < n_clusters; c += gridDim.x * blockDim.x) {

            if (BiprismOwnership) {
                const int action = prism_biprism_constraint_action_v1(d_biprism_ownership, n_atoms, h_clusters[c]);
                if (action < 0) {
                    prism_constraint_failure(d_constraint_window);
                    PRISM_PHYSICS_TRAP();
                    return;
                }
                if (action > 0) continue; // Excluded, not a successful solve/stamp.
            }
            if (!rattle_cluster_velocities<Compensated>(
                    positions, velocities, h_clusters[c], d_external_work, 100,coordinate_owner,precision_fault)) {
                prism_constraint_failure(d_constraint_window);
                PRISM_PHYSICS_TRAP();
                return;
            }
            // Existing coverage stamps are idempotent within one window.
            // The final post-bath RATTLE still executes and stamps phase 2.
            prism_constraint_success(d_constraint_window,
                execution_phase == MD_PHASE_RATTLE_BEFORE_OU_NP1 ? 2 : 0, c);
        }
        return;
    }

    // Save x_n for the constraint impulse and perform the full position drift.
    if (execution_phase == MD_PHASE_DRIFT_SAVE_XN) {
        if (tid < n_atoms) {
            if constexpr(Compensated) {
                if(!G::snapshot(positions_step_start,positions,tid,coordinate_owner,precision_fault))return;
            } else positions_step_start[tid] = positions[tid];
            if constexpr(Compensated) {
                if(!G::drift(positions,tid,dt,velocities[tid],coordinate_owner,precision_fault))return;
            } else {
            positions[tid].x += dt * velocities[tid].x;
            positions[tid].y += dt * velocities[tid].y;
            positions[tid].z += dt * velocities[tid].z;
            }
        }
        return;
    }

    // ========================================================================
    // PHASE 4: JOINT POSITION SHAKE CONSTRAINTS
    // ========================================================================

    if (execution_phase == MD_PHASE_SHAKE_POSITION) {
        for (int c = tid; c < n_clusters; c += gridDim.x * blockDim.x) {

            if (BiprismOwnership) {
                const int action = prism_biprism_constraint_action_v1(d_biprism_ownership, n_atoms, h_clusters[c]);
                if (action < 0) {
                    prism_constraint_failure(d_constraint_window);
                    PRISM_PHYSICS_TRAP();
                    return;
                }
                if (action > 0) continue; // Excluded, not a successful solve/stamp.
            }
            if(exact_nonbonded_exceptions_enabled < 0) {
                if(!prism_may_original_position_projection_v1(positions,h_clusters[c])) {
                    prism_constraint_failure(d_constraint_window);PRISM_PHYSICS_TRAP();return;
                }
                // No velocity correction: the complete impulse row remains
                // measured zero, while coordinates retain the real projection.
                prism_constraint_success(d_constraint_window,1,c);
                continue;
            }
            if (!shake_cluster_positions<Compensated>(
                    positions, positions_step_start, velocities, h_clusters[c], dt,
                    d_external_work, 100, timestep,coordinate_owner,precision_fault)) {
                prism_constraint_failure(d_constraint_window);
                PRISM_PHYSICS_TRAP();
                return;
            }
            prism_constraint_success(d_constraint_window, 1, c);
        }
        return;
    }

    // ========================================================================
    // PHASE 6: SECOND FORCE HALF-KICK + FINAL OU HALF-STEP
    // ========================================================================

    if (execution_phase == MD_PHASE_KICK_NP1_OU_HALF) {
        if (tid < n_atoms) {
            double* work_component = d_external_work != nullptr ? &d_external_work[tid] : nullptr;
            // The reversible branch must bypass both force scaling and the
            // ordinary 1000 kcal/mol/A safety clamp
            // (MAX_INTEGRATOR_FORCE_KCAL_MOL_A, applied in
            // apply_force_half_kick_with_intervention_work).  That magnitude
            // is the declared anchor of gearbox GATE 5, whose force bands are
            // fractions of it, so changing the constant retunes gate 5 as well
            // as this clamp.
            if (gamma == 0.0f) {
                bp_native_asc_measure_kick<Compensated>(coordinate_owner,precision_fault,d_asc_decision,tid,n_atoms,1,
                    reinterpret_cast<const float*>(&positions[tid]),reinterpret_cast<const float*>(&forces[tid]),
                    reinterpret_cast<const float*>(&velocities[tid]),1.0f,INFINITY,masses[tid],dt,AMBER_KCAL_TO_AMU_A2_PS2);
                apply_exact_force_half_kick(
                    velocities[tid], forces[tid], masses[tid], dt);
                return;
            }
            bp_native_asc_measure_kick<Compensated>(coordinate_owner,precision_fault,d_asc_decision,tid,n_atoms,1,
                reinterpret_cast<const float*>(&positions[tid]),reinterpret_cast<const float*>(&forces[tid]),
                reinterpret_cast<const float*>(&velocities[tid]),atom_lambda[tid],MAX_INTEGRATOR_FORCE_KCAL_MOL_A,masses[tid],dt,AMBER_KCAL_TO_AMU_A2_PS2);
            apply_force_half_kick_with_intervention_work(
                velocities[tid], forces[tid], atom_lambda[tid], masses[tid], dt,
                work_component);
            curandState local_rng = rng_states[tid];
            langevin_thermostat(
                velocities[tid], masses[tid], target_temp, gamma, 0.5f * dt,
                &local_rng, work_component);
            rng_states[tid] = local_rng;
        }
        return;
    }

    // ========================================================================
    // HOLOGRAPHIC EXCLUSION FIELD UPDATE (DISABLED IN THIS KERNEL)
    // ========================================================================

    int total_voxels = 0; // DISABLED: voxel work moved to nhs_voxel_step kernel

    for (int v = tid; v < total_voxels; v += gridDim.x * blockDim.x) {
        // Save previous water density
        water_density_prev[v] = water_density[v];

        // Voxel position
        int vz = v / (grid_dim * grid_dim);
        int vy = (v / grid_dim) % grid_dim;
        int vx = v % grid_dim;

        float3 voxel_center = make_float3(
            grid_origin.x + (vx + 0.5f) * grid_spacing,
            grid_origin.y + (vy + 0.5f) * grid_spacing,
            grid_origin.z + (vz + 0.5f) * grid_spacing
        );

        // Use warp matrix to only check nearby atoms (O(16) instead of O(N))
        WarpEntry entry = warp_matrix[v];
        float total_exclusion = 0.0f;
        float polar_field = 0.0f;

        // Only loop over atoms in this voxel's neighborhood (max 16)
        for (int i = 0; i < entry.n_atoms; i++) {
            int a = entry.atom_indices[i];
            if (a < 0 || a >= n_atoms) continue;

            float contrib = compute_exclusion_contribution(
                positions[a], voxel_center,
                atom_types[a], charges[a]
            );

            // EXPANDED UV EFFECT: Apply exclusion modifier to ALL atoms near excited aromatics
            // This expands the UV perturbation beyond just ring atoms to create
            // a larger "zone of influence" that the LIF neurons can detect.
            // The modifier considers ALL nearby excited aromatics with distance decay.
            if (n_aromatics > 0 && d_aromatic_centroids != nullptr) {
                float expanded_modifier = compute_expanded_exclusion_modifier(
                    positions[a],           // Position of this atom
                    d_aromatic_centroids,   // Positions of aromatic centroids
                    d_ring_normals,         // Ring normal directions
                    d_is_excited,           // Excitation flags
                    d_electronic_population,// Electronic populations
                    n_aromatics
                );
                contrib *= expanded_modifier;
            } else {
                // Fallback: original aromatic-only modifier
                int aromatic_idx = d_atom_to_aromatic[a];
                if (aromatic_idx >= 0 && aromatic_idx < n_aromatics) {
                    float excitation_modifier = get_exclusion_modifier(
                        aromatic_idx,
                        d_is_excited,
                        d_electronic_population
                    );
                    contrib *= excitation_modifier;
                }
            }

            total_exclusion += contrib * entry.atom_weights[i] * 4.0f;  // Scale by weight

            // Polar field from charged/polar atoms
            if (atom_types[a] == 1 || atom_types[a] == 2 || atom_types[a] == 3) {
                polar_field += contrib * 0.5f;
            }
        }

        // Clamp exclusion
        total_exclusion = fminf(1.0f, total_exclusion);
        exclusion_field[v] = total_exclusion;

        // Infer water density (temperature-dependent!)
        water_density[v] = infer_water_density(total_exclusion, polar_field, target_temp);
    }

    __syncthreads();

    // ========================================================================
    // PHASE 5: NEUROMORPHIC LIF OBSERVATION WITH DIRECT UV-LIF COUPLING
    // ========================================================================
    //
    // This phase combines standard water density-based LIF detection with
    // DIRECT UV-LIF coupling for enhanced UV-spike correlation:
    //
    // 1. Standard signal: Water density changes from exclusion field
    // 2. UV signal: Thermal wavefront + dewetting halo + cooperative effects
    //
    // The UV signal is computed from excited aromatics and injected directly
    // into the LIF membrane potential, creating DIRECT UV→spike coupling
    // that's phase-locked to UV bursts.

    float tau_mem = 0.1f;  // Membrane time constant (real decay per step)

    for (int v = tid; v < total_voxels; v += gridDim.x * blockDim.x) {
        // Refractory countdown: decrement if >0, only fire-able when ==0
        if (spike_grid_efp[v] > 0) spike_grid_efp[v]--;
        if (spike_grid[v] > 0) { spike_grid[v]--; continue; }
        float spike_intensity = 0.0f;

        // Compute voxel center position (needed for UV signal computation)
        int vz = v / (grid_dim * grid_dim);
        int vy = (v / grid_dim) % grid_dim;
        int vx = v % grid_dim;
        float3 voxel_center = make_float3(
            grid_origin.x + (vx + 0.5f) * grid_spacing,
            grid_origin.y + (vy + 0.5f) * grid_spacing,
            grid_origin.z + (vz + 0.5f) * grid_spacing
        );

        // ====================================================================
        // UV-LIF SIGNAL COMPUTATION (Direct UV → LIF coupling)
        // ====================================================================
        // This computes a UV-induced signal that's injected directly into the
        // LIF membrane potential, creating strong UV-spike correlation.
        //
        // Mechanisms:
        // - Thermal wavefront propagation from excited aromatics
        // - Dewetting halo effect (inner attraction + outer contrast)
        // - Cooperative enhancement for multiple nearby excited aromatics
        // - Temporal derivative amplification for UV-specific signals

        float uv_signal = 0.0f;
        int n_nearby_excited = 0;
        float min_distance_to_excited = 1000.0f;
        int closest_excited_idx = -1;
        if (n_aromatics > 0) {
            // ================================================================
            // SIMPLE DIRECT UV SIGNAL: Count nearby excited aromatics
            // ================================================================
            // This is a simplified but robust UV signal that directly correlates
            // with UV bursts. It counts how many excited aromatics are near this
            // voxel and scales the signal by vibrational energy.

            // VERY TIGHT detection radius for spatial localization
            // Only voxels essentially AT aromatic positions should trigger UV spikes
            const float UV_DETECTION_RADIUS = 4.0f;   // Å - very tight
            const float UV_DIRECT_STRENGTH = 0.8f;    // Strong signal for close voxels

            n_nearby_excited = 0;
            float total_vib_energy = 0.0f;
            min_distance_to_excited = 1000.0f;  // Track closest excited aromatic
            closest_excited_idx = -1;  // Index of closest excited aromatic

            for (int a = 0; a < n_aromatics; a++) {
                if (!d_is_excited[a]) continue;

                // Get aromatic centroid position (d_aromatic_centroids is const float3*)
                if (d_aromatic_centroids == nullptr) {
                    continue;  // Skip if no centroids
                }
                float3 arom_pos = d_aromatic_centroids[a];

                // Distance check
                float dx = voxel_center.x - arom_pos.x;
                float dy = voxel_center.y - arom_pos.y;
                float dz = voxel_center.z - arom_pos.z;
                float dist_sq = dx*dx + dy*dy + dz*dz;
                float dist = sqrtf(dist_sq);

                if (dist < UV_DETECTION_RADIUS) {
                    n_nearby_excited++;
                    total_vib_energy += d_vibrational_energy[a];
                    if (dist < min_distance_to_excited) {
                        min_distance_to_excited = dist;
                        closest_excited_idx = a;
                    }
                }
            }

            // DEBUG: Print UV signal computation (once per 10000 voxels, first block only)
            #ifdef DEBUG_UV_LIF
            if (blockIdx.x == 0 && threadIdx.x == 0 && v == 0 && timestep % 10 == 0) {
                printf("[UV-LIF DEBUG] ts=%d n_arom=%d centroid[0]=(%.2f,%.2f,%.2f) voxel=(%.2f,%.2f,%.2f)\n",
                       timestep, n_aromatics,
                       d_aromatic_centroids != nullptr ? d_aromatic_centroids[0].x : -1.0f,
                       d_aromatic_centroids != nullptr ? d_aromatic_centroids[0].y : -1.0f,
                       d_aromatic_centroids != nullptr ? d_aromatic_centroids[0].z : -1.0f,
                       voxel_center.x, voxel_center.y, voxel_center.z);
            }
            #endif

            // Compute simple UV signal
            if (n_nearby_excited > 0) {
                // Scale by number of aromatics and their energy
                float energy_factor = total_vib_energy / (n_nearby_excited * 3.0f);  // Normalize
                energy_factor = fminf(energy_factor, 1.0f);

                // Cooperative boost for multiple aromatics
                float coop_boost = 1.0f + 0.3f * (n_nearby_excited - 1);

                uv_signal = UV_DIRECT_STRENGTH * energy_factor * coop_boost;

                // Additional boost during active UV burst
                if (uv_burst_active) {
                    uv_signal *= 2.0f;
                }
            }

            // ================================================================
            // ADVANCED UV-LIF SIGNAL (thermal wavefront + halo)
            // ================================================================
            // Add the sophisticated physics-based signal on top of the direct signal
            if (d_aromatic_centroids != nullptr && d_uv_signal_prev != nullptr) {
                float prev_signal = d_uv_signal_prev[v];

                float advanced_signal = compute_uv_lif_signal(
                    voxel_center,
                    d_aromatic_centroids,
                    d_ring_normals,
                    d_is_excited,
                    d_electronic_population,
                    d_vibrational_energy,
                    d_time_since_excitation,
                    n_aromatics,
                    dt,
                    prev_signal
                );

                uv_signal += advanced_signal;
                d_uv_signal_prev[v] = uv_signal;
            }

            // INJECT combined UV signal into LIF membrane potential
            if (uv_signal > 0.0f) {
                lif_potential[v] += uv_signal;
            }

            // ================================================================
            // DIRECT UV SPIKE TRIGGER
            // ================================================================
            // During UV bursts, if UV signal is strong enough, create a spike
            // IMMEDIATELY without waiting for slow LIF membrane accumulation.
            // This creates strong UV-spike correlation.
            //
            // The direct spike threshold is much lower than LIF_THRESHOLD,
            // ensuring UV-induced spikes happen DURING/shortly after bursts.

            // INCREASED threshold to reduce noise, REQUIRE proximity to aromatic
            const float DIRECT_UV_SPIKE_THRESHOLD = 0.3f;   // Moderate threshold
            const float MAX_SPIKE_DISTANCE = 4.0f;  // Å - within 4Å of aromatic centroid

            // Check if this voxel CONTAINS any aromatic atoms
            // This ensures UV spikes happen AT aromatic residues, not just near them
            const int voxel_n_atoms = warp_matrix[v].n_atoms;
            bool voxel_has_aromatic_atom = false;
            for (int wi = 0; wi < voxel_n_atoms && !voxel_has_aromatic_atom; wi++) {
                int atom_idx = warp_matrix[v].atom_indices[wi];
                if (atom_idx >= 0 && atom_idx < n_atoms) {
                    // Check if this atom belongs to an aromatic (d_atom_to_aromatic >= 0)
                    if (d_atom_to_aromatic[atom_idx] >= 0) {
                        voxel_has_aromatic_atom = true;
                    }
                }
            }

            // Only trigger UV spike if:
            // 1. Nearby excited aromatics (n_nearby_excited > 0)
            // 2. UV signal above threshold
            // 3. Voxel has atoms mapped to it
            // 4. Voxel is close to an excited aromatic
            // 5. CRITICAL: Voxel CONTAINS an aromatic atom (ensures spike is AT aromatic location)
            if (n_nearby_excited > 0 &&
                uv_signal > DIRECT_UV_SPIKE_THRESHOLD &&
                voxel_n_atoms > 0 &&
                min_distance_to_excited < MAX_SPIKE_DISTANCE) {
                spike_grid[v] = REFRACTORY_STEPS;
                spike_intensity = uv_signal;  // Use UV signal as intensity

                // Signal preservation
                update_signal_preservation(v, timestep, 1,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[v], residue_ids, n_atoms, residue_step_causal);

                // Extract closest aromatic metadata for enhanced spike event
                int _arom_type = (closest_excited_idx >= 0) ? d_aromatic_type[closest_excited_idx] : -1;
                int _arom_res = -1;
                if (closest_excited_idx >= 0) {
                    for (int wi = 0; wi < warp_matrix[v].n_atoms && _arom_res < 0; wi++) {
                        int ai = warp_matrix[v].atom_indices[wi];
                        if (ai >= 0 && ai < n_atoms && d_atom_to_aromatic[ai] == closest_excited_idx) {
                            _arom_res = residue_ids[ai];
                        }
                    }
                }
                float _vib_e = (closest_excited_idx >= 0) ? d_vibrational_energy[closest_excited_idx] : 0.0f;

                int spike_idx = atomicAdd(spike_count, 1);
                if (spike_idx < max_spikes) {
                    capture_spike_event(
                        spike_events[spike_idx],
                        timestep,
                        v,
                        voxel_center,
                        spike_intensity,
                        warp_matrix[v],
                        residue_ids,
                        1,                      // spike_source = UV
                        uv_wavelength_nm,       // wavelength
                        _arom_type,             // aromatic_type
                        _arom_res,              // aromatic_residue_id
                        water_density[v],       // water_density
                        _vib_e,                 // vibrational_energy
                        n_nearby_excited,       // n_nearby_excited
                        fabsf(water_density[v] - water_density_prev[v]),  // wd_change
                        d_protocol->current_phase_bits  // CCNS phase
                    );
                }

                // Reset membrane potential after spike
                lif_potential[v] = LIF_RESET;
            }
        }

        // ====================================================================
        // STANDARD LIF UPDATE (Water density-based detection)
        // ====================================================================
        // This detects thermal/conformational changes through water density.
        // UV-induced spikes are handled above; this catches everything else.

        // Skip if we already spiked from UV
        // ALSO skip during active UV burst - only UV-triggered spikes during bursts
        // This ensures clean UV-spike correlation for analysis
        if (spike_grid[v] == 0 && !uv_burst_active) {
            bool spike = lif_neuron_update(
                lif_potential[v],
                water_density[v],
                water_density_prev[v],
                tau_mem,
                dt,
                LIF_THRESHOLD,
                spike_intensity  // Captures pre-reset membrane potential
            );

            if (spike) {
                spike_grid[v] = REFRACTORY_STEPS;

                // Signal preservation
                int lif_source = (n_nearby_excited > 0) ? 1 : 2;
                update_signal_preservation(v, timestep, lif_source,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[v], residue_ids, n_atoms, residue_step_causal);

                // Capture spike event with proper intensity
                int spike_idx = atomicAdd(spike_count, 1);
                if (spike_idx < max_spikes) {
                    // If voxel had UV contribution, carry aromatic metadata
                    int lif_atype = -1;
                    int lif_ares = -1;
                    float lif_wl = 0.0f;
                    float lif_vibe = 0.0f;
                    if (n_nearby_excited > 0 && closest_excited_idx >= 0) {
                        lif_atype = d_aromatic_type[closest_excited_idx];
                        lif_wl = uv_wavelength_nm;
                        lif_vibe = d_vibrational_energy[closest_excited_idx];
                        // Find residue ID for closest aromatic
                        for (int wi = 0; wi < warp_matrix[v].n_atoms && lif_ares < 0; wi++) {
                            int ai = warp_matrix[v].atom_indices[wi];
                            if (ai >= 0 && ai < n_atoms && d_atom_to_aromatic[ai] == closest_excited_idx) {
                                lif_ares = residue_ids[ai];
                            }
                        }
                    }
                    capture_spike_event(
                        spike_events[spike_idx],
                        timestep,
                        v,
                        voxel_center,
                        spike_intensity,
                        warp_matrix[v],
                        residue_ids,
                        lif_source,
                        lif_wl,
                        lif_atype,
                        lif_ares,
                        water_density[v],
                        lif_vibe,
                        n_nearby_excited,
                        fabsf(water_density[v] - water_density_prev[v]),  // wd_change
                        d_protocol->current_phase_bits  // CCNS phase
                    );
                }
            }
        }
    }

    __syncthreads();

    // ========================================================================
    // ====================================================================
    // ELECTROSTATIC FLUX PROBE (EFP) - Polar Binding Site Detection
    // ====================================================================
    for (int efp_v = tid; efp_v < total_voxels; efp_v += gridDim.x * blockDim.x) {
        if (warp_matrix[efp_v].n_atoms == 0) continue;

        int efp_vz = efp_v / (grid_dim * grid_dim);
        int efp_vy = (efp_v / grid_dim) % grid_dim;
        int efp_vx = efp_v % grid_dim;
        float3 efp_center = make_float3(
            grid_origin_x + (efp_vx + 0.5f) * grid_spacing,
            grid_origin_y + (efp_vy + 0.5f) * grid_spacing,
            grid_origin_z + (efp_vz + 0.5f) * grid_spacing
        );

        float phi = 0.0f;
        int n_charged_nearby = 0;

        for (int wi = 0; wi < warp_matrix[efp_v].n_atoms; wi++) {
            int ai = warp_matrix[efp_v].atom_indices[wi];
            if (ai < 0 || ai >= n_atoms) continue;
            float q = charges[ai];
            if (fabsf(q) < 0.15f) continue;

            float3 ap = positions[ai];
            float dx = efp_center.x - ap.x;
            float dy = efp_center.y - ap.y;
            float dz = efp_center.z - ap.z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);

            if (dist > 0.5f && dist < 8.0f) {
                float eps_r = fmaxf(4.0f * dist, 4.0f);
                phi += q / (eps_r * dist);
                n_charged_nearby++;
            }
        }

        // Water-electrostatic coupling
        float wd_change = fabsf(water_density[efp_v] - water_density_prev[efp_v]);
        float polar_water_signal = fabsf(phi) * wd_change * 40.0f;

        float phi_prev = efp_potential_prev[efp_v];
        efp_potential[efp_v] = phi;

        float flux = fabsf(phi - phi_prev);
        float polar_signal = flux * 150.0f + polar_water_signal;

        if (n_charged_nearby >= 1 && spike_grid_efp[efp_v] == 0) {
            const float EFP_TAU = 0.5f;
            const float EFP_THRESHOLD = 0.15f;
            float efp_decay = expf(-dt / EFP_TAU);
            efp_lif_potential[efp_v] = efp_decay * efp_lif_potential[efp_v] + polar_signal;

            if (efp_lif_potential[efp_v] > EFP_THRESHOLD) {
                spike_grid_efp[efp_v] = REFRACTORY_STEPS;
                float polar_intensity = efp_lif_potential[efp_v];
                efp_lif_potential[efp_v] = LIF_RESET;

                // Signal preservation (EFP spike)
                update_signal_preservation(efp_v, timestep, 3,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[efp_v], residue_ids, n_atoms, residue_step_causal);

                int si = atomicAdd(spike_count, 1);
                if (si < max_spikes) {
                    int polar_type = (phi > 0.0f) ? 5 : 6;
                    capture_spike_event(
                        spike_events[si], timestep, efp_v, efp_center,
                        polar_intensity, warp_matrix[efp_v], residue_ids,
                        3, 0.0f, polar_type, -1,
                        water_density[efp_v], flux, n_charged_nearby,
                        wd_change,  // already computed above
                        d_protocol->current_phase_bits  // CCNS phase
                    );
                }
            }
        }
        efp_potential_prev[efp_v] = phi;
    }
    __syncthreads();

    // Endpoint interventions are intentionally stream-split by the host. After
    // phase 6 retires, the host launches UV impulse, velocity safety clamp,
    // final RATTLE, then the deterministic non-Hamiltonian-work reduction. Each
    // dependency is therefore a real kernel boundary rather than a block-local
    // __syncthreads() masquerading as a grid-wide barrier.
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_fused_step(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision
) {
    prism_amber_phase_body_v1<-1>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}

// Explicit phase0; same ABI109 and same shared expressions.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_phase_force_n_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision

) {
    prism_amber_phase_body_v1<0>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}

// Explicit phase1; same ABI109 and same shared expressions.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_phase_ou_half_kick_n_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision

) {
    prism_amber_phase_body_v1<1>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}

// Explicit phase2; same ABI109 and same shared expressions.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_phase_rattle_pre_drift_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision

) {
    prism_amber_phase_body_v1<2>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}

// Explicit phase3; same ABI109 and same shared expressions.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_phase_drift_save_xn_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision

) {
    prism_amber_phase_body_v1<3>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}

// Explicit phase4; same ABI109 and same shared expressions.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_phase_shake_position_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision

) {
    prism_amber_phase_body_v1<4>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}

// Explicit phase5; same ABI109 and same shared expressions.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_phase_force_np1_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision

) {
    prism_amber_phase_body_v1<5>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}

// Explicit phase6; same ABI109 and same shared expressions.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_phase_kick_np1_ou_half_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision

) {
    prism_amber_phase_body_v1<6>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}

// Explicit phase10; same ABI109 and same shared expressions.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_phase_ou_half_n_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision

) {
    prism_amber_phase_body_v1<10>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}

// Explicit phase11; same ABI109 and same shared expressions.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_phase_rattle_after_ou_n_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision

) {
    prism_amber_phase_body_v1<11>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}

// Explicit phase12; same ABI109 and same shared expressions.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_phase_kick_n_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision

) {
    prism_amber_phase_body_v1<12>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}

// Explicit phase13; same ABI109 and same shared expressions.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_phase_kick_np1_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision

) {
    prism_amber_phase_body_v1<13>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}

// Explicit phase14; same ABI109 and same shared expressions.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_phase_rattle_before_ou_np1_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision

) {
    prism_amber_phase_body_v1<14>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}

// Explicit phase15; same ABI109 and same shared expressions.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_amber_phase_ou_half_np1_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision

) {
    prism_amber_phase_body_v1<15>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision
    );
}


// Preserve the charge parameters used by FORCE_NP1 before the UV pump changes
// them. Buffers are engine-owned and reused by every captured physical body.
// Explicit nondefault Biprism/AMBER phase exports. The first109 arguments
// retain native order; argument110 is the immutable per-atom ownership mask.
// Force membership/formulas are unchanged. Caller must bind the new profile,
// pre-core-scatter x_n snapshot, mask lifetime and policy-specific coverage.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_force_n_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership

) {
    prism_amber_phase_body_v1<0, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_ou_half_kick_n_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership

) {
    prism_amber_phase_body_v1<1, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_rattle_pre_drift_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership

) {
    prism_amber_phase_body_v1<2, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_drift_save_xn_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership

) {
    prism_amber_phase_body_v1<3, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_shake_position_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership

) {
    prism_amber_phase_body_v1<4, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_force_np1_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership

) {
    prism_amber_phase_body_v1<5, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_kick_np1_ou_half_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership

) {
    prism_amber_phase_body_v1<6, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_ou_half_n_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership

) {
    prism_amber_phase_body_v1<10, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_rattle_after_ou_n_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership

) {
    prism_amber_phase_body_v1<11, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_kick_n_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership

) {
    prism_amber_phase_body_v1<12, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_kick_np1_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership

) {
    prism_amber_phase_body_v1<13, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_rattle_before_ou_np1_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership

) {
    prism_amber_phase_body_v1<14, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_ou_half_np1_v1(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format: 1-2/1-3 fully excluded
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,     // CSR format: 1-4 scaled pairs
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    const WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,

    // PRISM-TWIN v3.0: GPU-resident protocol state (replaces temperature, UV, dt, gamma, timestep scalars)
    const ProtocolState* __restrict__ d_protocol,

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Static simulation parameters (safe to bake in CUDA Graph)
    float cutoff,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update

    // M1.2.17 — Per-atom potential-energy components buffer.
    // [n_atoms] f64.  Zeroed each step by the host before launch; the
    // live deterministic-gather force phase accumulates this atom's V
    // splits (V_b/2 per bonded atom; V_a/3 per angle atom; V_d/4 per
    // dihedral atom; V_pair/2 per non-bonded pair atom) into a register
    // and then DESTRUCTIVELY STORES the row (pe_components[tid] =
    // local_pe) — one writer per atom, no atomics.  The only
    // atomicAdd_f64 path inside this kernel is the legacy atomic force
    // branch below, which traps before it runs.  Nodes launched after
    // this store on the same stream (the lambda dual-endpoint
    // Hamiltonian, the ASC steering kernel) do accumulate on top of the
    // stored row.  The captured CUB DeviceReduce::Sum node consumes it and
    // writes the system-wide V into adj.d_potential_energy at offset
    // 112.  Pass nullptr to disable PE accumulation.
    double* pe_components,

    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // [n_atoms] f64. Zeroed by the host before launch. The fused integration
    // phase writes per-atom clamp work directly to d_external_work[tid], the
    // ordered UV phase writes per-neighbor kick work directly to
    // d_external_work[neighbor], and a fixed-order reducer writes the scalar
    // total to d_external_work[0] for the SFA stability fuse.
    // Pass nullptr to disable W_ext accumulation (legacy / non-V2).
    double* d_external_work,

    // Deterministic atom-owned bonded incidence lists. When present, the
    // AMBER force phase uses per-atom gather reductions and direct writes
    // instead of cross-block f32 atomicAdd into forces.
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,

    // Grand-canonical multisite lambda-fragment force plane. atom_slot=-1 is
    // an ordinary atom. dU/dlambda is atom-owned for deterministic reduction.
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,

    // Step-start positions used by the SHAKE/RATTLE position impulse.
    float3* positions_step_start,

    // Kernel-boundary phase split. A regular multi-block CUDA kernel has no
    // grid-wide barrier, so every dependent stage must retire as a distinct
    // launch. See MD_PHASE_* above. Endpoint UV, velocity safety clamp, and
    // final RATTLE are separate kernels.
    int execution_phase,

    // P5 Step 1 — per-aromatic MEAN ground-state charge over that aromatic's
    // mapped atom set. Enables a CHARGE-NEUTRAL excited-state rescale; see
    // uv_excited_charge above. Null on legacy paths, which degrades to the
    // historical multiplicative form rather than to a zero mean.
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership

) {
    prism_amber_phase_body_v1<15, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership
    );
}

// Compensated profile: original110 arguments, then native owner96 and terminal fault sink.
extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_force_n_compensated_v1(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,
    const HCluster* h_clusters, int n_clusters,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    const WarpEntry* warp_matrix,
    const UVTarget* uv_targets, int n_uv_targets,
    const ProtocolState* __restrict__ d_protocol,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const float* d_ground_state_charges,
    const int* d_atom_to_aromatic,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const float3* d_aromatic_centroids,
    float* d_uv_signal_prev,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float cutoff,
    curandState* rng_states,
    const int* neighbor_list,
    const int* n_neighbors,
    int use_neighbor_list,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int* spike_grid_efp,
    const float* __restrict__ atom_lambda,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    int* residue_step_causal,
    double* pe_components,
    double* d_external_work,
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,
    float3* positions_step_start,
    int execution_phase,
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    prism_amber_phase_body_v1<0, true, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership, coordinate_owner, precision_fault
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_ou_half_kick_n_compensated_v1(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,
    const HCluster* h_clusters, int n_clusters,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    const WarpEntry* warp_matrix,
    const UVTarget* uv_targets, int n_uv_targets,
    const ProtocolState* __restrict__ d_protocol,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const float* d_ground_state_charges,
    const int* d_atom_to_aromatic,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const float3* d_aromatic_centroids,
    float* d_uv_signal_prev,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float cutoff,
    curandState* rng_states,
    const int* neighbor_list,
    const int* n_neighbors,
    int use_neighbor_list,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int* spike_grid_efp,
    const float* __restrict__ atom_lambda,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    int* residue_step_causal,
    double* pe_components,
    double* d_external_work,
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,
    float3* positions_step_start,
    int execution_phase,
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    prism_amber_phase_body_v1<1, true, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership, coordinate_owner, precision_fault
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_rattle_pre_drift_compensated_v1(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,
    const HCluster* h_clusters, int n_clusters,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    const WarpEntry* warp_matrix,
    const UVTarget* uv_targets, int n_uv_targets,
    const ProtocolState* __restrict__ d_protocol,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const float* d_ground_state_charges,
    const int* d_atom_to_aromatic,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const float3* d_aromatic_centroids,
    float* d_uv_signal_prev,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float cutoff,
    curandState* rng_states,
    const int* neighbor_list,
    const int* n_neighbors,
    int use_neighbor_list,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int* spike_grid_efp,
    const float* __restrict__ atom_lambda,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    int* residue_step_causal,
    double* pe_components,
    double* d_external_work,
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,
    float3* positions_step_start,
    int execution_phase,
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    prism_amber_phase_body_v1<2, true, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership, coordinate_owner, precision_fault
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_drift_save_xn_compensated_v1(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,
    const HCluster* h_clusters, int n_clusters,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    const WarpEntry* warp_matrix,
    const UVTarget* uv_targets, int n_uv_targets,
    const ProtocolState* __restrict__ d_protocol,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const float* d_ground_state_charges,
    const int* d_atom_to_aromatic,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const float3* d_aromatic_centroids,
    float* d_uv_signal_prev,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float cutoff,
    curandState* rng_states,
    const int* neighbor_list,
    const int* n_neighbors,
    int use_neighbor_list,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int* spike_grid_efp,
    const float* __restrict__ atom_lambda,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    int* residue_step_causal,
    double* pe_components,
    double* d_external_work,
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,
    float3* positions_step_start,
    int execution_phase,
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    prism_amber_phase_body_v1<3, true, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership, coordinate_owner, precision_fault
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_shake_position_compensated_v1(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,
    const HCluster* h_clusters, int n_clusters,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    const WarpEntry* warp_matrix,
    const UVTarget* uv_targets, int n_uv_targets,
    const ProtocolState* __restrict__ d_protocol,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const float* d_ground_state_charges,
    const int* d_atom_to_aromatic,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const float3* d_aromatic_centroids,
    float* d_uv_signal_prev,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float cutoff,
    curandState* rng_states,
    const int* neighbor_list,
    const int* n_neighbors,
    int use_neighbor_list,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int* spike_grid_efp,
    const float* __restrict__ atom_lambda,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    int* residue_step_causal,
    double* pe_components,
    double* d_external_work,
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,
    float3* positions_step_start,
    int execution_phase,
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    prism_amber_phase_body_v1<4, true, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership, coordinate_owner, precision_fault
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_force_np1_compensated_v1(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,
    const HCluster* h_clusters, int n_clusters,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    const WarpEntry* warp_matrix,
    const UVTarget* uv_targets, int n_uv_targets,
    const ProtocolState* __restrict__ d_protocol,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const float* d_ground_state_charges,
    const int* d_atom_to_aromatic,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const float3* d_aromatic_centroids,
    float* d_uv_signal_prev,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float cutoff,
    curandState* rng_states,
    const int* neighbor_list,
    const int* n_neighbors,
    int use_neighbor_list,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int* spike_grid_efp,
    const float* __restrict__ atom_lambda,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    int* residue_step_causal,
    double* pe_components,
    double* d_external_work,
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,
    float3* positions_step_start,
    int execution_phase,
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    prism_amber_phase_body_v1<5, true, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership, coordinate_owner, precision_fault
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_kick_np1_ou_half_compensated_v1(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,
    const HCluster* h_clusters, int n_clusters,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    const WarpEntry* warp_matrix,
    const UVTarget* uv_targets, int n_uv_targets,
    const ProtocolState* __restrict__ d_protocol,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const float* d_ground_state_charges,
    const int* d_atom_to_aromatic,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const float3* d_aromatic_centroids,
    float* d_uv_signal_prev,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float cutoff,
    curandState* rng_states,
    const int* neighbor_list,
    const int* n_neighbors,
    int use_neighbor_list,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int* spike_grid_efp,
    const float* __restrict__ atom_lambda,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    int* residue_step_causal,
    double* pe_components,
    double* d_external_work,
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,
    float3* positions_step_start,
    int execution_phase,
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    prism_amber_phase_body_v1<6, true, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership, coordinate_owner, precision_fault
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_ou_half_n_compensated_v1(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,
    const HCluster* h_clusters, int n_clusters,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    const WarpEntry* warp_matrix,
    const UVTarget* uv_targets, int n_uv_targets,
    const ProtocolState* __restrict__ d_protocol,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const float* d_ground_state_charges,
    const int* d_atom_to_aromatic,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const float3* d_aromatic_centroids,
    float* d_uv_signal_prev,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float cutoff,
    curandState* rng_states,
    const int* neighbor_list,
    const int* n_neighbors,
    int use_neighbor_list,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int* spike_grid_efp,
    const float* __restrict__ atom_lambda,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    int* residue_step_causal,
    double* pe_components,
    double* d_external_work,
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,
    float3* positions_step_start,
    int execution_phase,
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    prism_amber_phase_body_v1<10, true, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership, coordinate_owner, precision_fault
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_rattle_after_ou_n_compensated_v1(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,
    const HCluster* h_clusters, int n_clusters,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    const WarpEntry* warp_matrix,
    const UVTarget* uv_targets, int n_uv_targets,
    const ProtocolState* __restrict__ d_protocol,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const float* d_ground_state_charges,
    const int* d_atom_to_aromatic,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const float3* d_aromatic_centroids,
    float* d_uv_signal_prev,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float cutoff,
    curandState* rng_states,
    const int* neighbor_list,
    const int* n_neighbors,
    int use_neighbor_list,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int* spike_grid_efp,
    const float* __restrict__ atom_lambda,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    int* residue_step_causal,
    double* pe_components,
    double* d_external_work,
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,
    float3* positions_step_start,
    int execution_phase,
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    prism_amber_phase_body_v1<11, true, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership, coordinate_owner, precision_fault
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_kick_n_compensated_v1(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,
    const HCluster* h_clusters, int n_clusters,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    const WarpEntry* warp_matrix,
    const UVTarget* uv_targets, int n_uv_targets,
    const ProtocolState* __restrict__ d_protocol,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const float* d_ground_state_charges,
    const int* d_atom_to_aromatic,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const float3* d_aromatic_centroids,
    float* d_uv_signal_prev,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float cutoff,
    curandState* rng_states,
    const int* neighbor_list,
    const int* n_neighbors,
    int use_neighbor_list,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int* spike_grid_efp,
    const float* __restrict__ atom_lambda,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    int* residue_step_causal,
    double* pe_components,
    double* d_external_work,
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,
    float3* positions_step_start,
    int execution_phase,
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    prism_amber_phase_body_v1<12, true, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership, coordinate_owner, precision_fault
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_kick_np1_compensated_v1(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,
    const HCluster* h_clusters, int n_clusters,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    const WarpEntry* warp_matrix,
    const UVTarget* uv_targets, int n_uv_targets,
    const ProtocolState* __restrict__ d_protocol,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const float* d_ground_state_charges,
    const int* d_atom_to_aromatic,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const float3* d_aromatic_centroids,
    float* d_uv_signal_prev,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float cutoff,
    curandState* rng_states,
    const int* neighbor_list,
    const int* n_neighbors,
    int use_neighbor_list,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int* spike_grid_efp,
    const float* __restrict__ atom_lambda,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    int* residue_step_causal,
    double* pe_components,
    double* d_external_work,
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,
    float3* positions_step_start,
    int execution_phase,
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    prism_amber_phase_body_v1<13, true, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership, coordinate_owner, precision_fault
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_rattle_before_ou_np1_compensated_v1(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,
    const HCluster* h_clusters, int n_clusters,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    const WarpEntry* warp_matrix,
    const UVTarget* uv_targets, int n_uv_targets,
    const ProtocolState* __restrict__ d_protocol,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const float* d_ground_state_charges,
    const int* d_atom_to_aromatic,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const float3* d_aromatic_centroids,
    float* d_uv_signal_prev,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float cutoff,
    curandState* rng_states,
    const int* neighbor_list,
    const int* n_neighbors,
    int use_neighbor_list,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int* spike_grid_efp,
    const float* __restrict__ atom_lambda,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    int* residue_step_causal,
    double* pe_components,
    double* d_external_work,
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,
    float3* positions_step_start,
    int execution_phase,
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    prism_amber_phase_body_v1<14, true, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership, coordinate_owner, precision_fault
    );
}

extern "C" __global__ void __launch_bounds__(256, 1) nhs_biprism_amber_phase_ou_half_np1_compensated_v1(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ pairs14_list,
    const int* __restrict__ pairs14_offsets,
    const float* __restrict__ pairs14_charge_product,
    const float* __restrict__ pairs14_sigma,
    const float* __restrict__ pairs14_epsilon,
    int exact_nonbonded_exceptions_enabled,
    const HCluster* h_clusters, int n_clusters,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    const WarpEntry* warp_matrix,
    const UVTarget* uv_targets, int n_uv_targets,
    const ProtocolState* __restrict__ d_protocol,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const float* d_ground_state_charges,
    const int* d_atom_to_aromatic,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const float3* d_aromatic_centroids,
    float* d_uv_signal_prev,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float cutoff,
    curandState* rng_states,
    const int* neighbor_list,
    const int* n_neighbors,
    int use_neighbor_list,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int* spike_grid_efp,
    const float* __restrict__ atom_lambda,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    int* residue_step_causal,
    double* pe_components,
    double* d_external_work,
    const int* atom_bond_list,
    const int* atom_bond_count,
    const int* atom_bond_position,
    int max_bonds_per_atom,
    const int* atom_angle_list,
    const int* atom_angle_count,
    const int* atom_angle_position,
    int max_angles_per_atom,
    const int* atom_dihedral_list,
    const int* atom_dihedral_count,
    const int* atom_dihedral_position,
    int max_dihedrals_per_atom,
    const int* lambda_atom_slot,
    const float* lambda_slot_values,
    const float* lambda_current_charges,
    const float* lambda_endpoint_a_charges,
    const float* lambda_endpoint_b_charges,
    const float* lambda_charge_derivatives,
    const int* lambda_endpoint_occupancy_masks,
    int n_lambda_slots,
    float lambda_alpha_lj,
    float lambda_alpha_q_a2,
    float lambda_softcore_power,
    float lambda_coupling_power,
    int lambda_v2_ewald_enabled,
    float lambda_pme_beta_a_inv,
    float lambda_box_x,
    float lambda_box_y,
    float lambda_box_z,
    double* lambda_dudl_atom,
    double* lambda_energy_atom,
    float3* positions_step_start,
    int execution_phase,
    const float* d_aromatic_charge_set_mean,
    PrismConstraintWindowV1* d_constraint_window,
    PrismAscDeviceV1* d_asc_decision,
    const uint8_t* d_biprism_ownership,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    prism_amber_phase_body_v1<15, true, true>(
        positions, velocities, forces, masses, charges, atom_types, residue_ids, n_atoms, bonds, n_bonds, angles, n_angles, dihedrals, n_dihedrals, lj_params, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, pairs14_charge_product, pairs14_sigma, pairs14_epsilon, exact_nonbonded_exceptions_enabled, h_clusters, n_clusters, exclusion_field, water_density, water_density_prev, lif_potential, spike_grid, grid_origin_x, grid_origin_y, grid_origin_z, grid_spacing, grid_dim, warp_matrix, uv_targets, n_uv_targets, d_protocol, d_is_excited, d_time_since_excitation, d_electronic_population, d_vibrational_energy, d_franck_condon_progress, d_ground_state_charges, d_atom_to_aromatic, d_aromatic_type, d_ring_normals, d_aromatic_centroids, d_uv_signal_prev, d_aromatic_neighbors, n_aromatics, spike_events, spike_count, max_spikes, cutoff, rng_states, neighbor_list, n_neighbors, use_neighbor_list, efp_potential, efp_potential_prev, efp_lif_potential, spike_grid_efp, atom_lambda, voxel_hit_grid, last_uv_step, coupled_spike_grid, primary_residue_id, primary_residue_count, residue_step_causal, pe_components, d_external_work, atom_bond_list, atom_bond_count, atom_bond_position, max_bonds_per_atom, atom_angle_list, atom_angle_count, atom_angle_position, max_angles_per_atom, atom_dihedral_list, atom_dihedral_count, atom_dihedral_position, max_dihedrals_per_atom, lambda_atom_slot, lambda_slot_values, lambda_current_charges, lambda_endpoint_a_charges, lambda_endpoint_b_charges, lambda_charge_derivatives, lambda_endpoint_occupancy_masks, n_lambda_slots, lambda_alpha_lj, lambda_alpha_q_a2, lambda_softcore_power, lambda_coupling_power, lambda_v2_ewald_enabled, lambda_pme_beta_a_inv, lambda_box_x, lambda_box_y, lambda_box_z, lambda_dudl_atom, lambda_energy_atom, positions_step_start, execution_phase, d_aromatic_charge_set_mean, d_constraint_window, d_asc_decision, d_biprism_ownership, coordinate_owner, precision_fault
    );
}

extern "C" __global__ void nhs_uv_charge_state_snapshot(
    const int* is_excited, const float* population,
    int* is_excited_before, float* population_before, int n_aromatics
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_aromatics) {
        is_excited_before[i] = is_excited[i];
        population_before[i] = population[i];
    }
}

__device__ __forceinline__ float uv_charge_for_state(
    int atom, const float* charges, const float* ground_charges,
    const int* atom_to_aromatic, const int* aromatic_type, const float* charge_set_mean,
    const int* is_excited, const float* population, float& scale
) {
    scale = 1.0f;
    const int arom = atom_to_aromatic[atom];
    if (arom < 0 || !is_excited[arom]) return charges[atom];
    float ratio_sqrt;
    switch (aromatic_type[arom]) {
        case 0: ratio_sqrt = TRP_DIPOLE_RATIO_SQRT; break;
        case 1: ratio_sqrt = TYR_DIPOLE_RATIO_SQRT; break;
        case 2: ratio_sqrt = PHE_DIPOLE_RATIO_SQRT; break;
        default: ratio_sqrt = 1.0f;
    }
    scale = 1.0f + (ratio_sqrt - 1.0f) * population[arom];
    return uv_excited_charge(ground_charges[atom], charge_set_mean, arom, scale);
}

// Signed existing nonbonded mode: -1 historical May, 0 explicit partial, +1 exact V2.
// Charge-only difference of the same Hamiltonians used above. In legacy V1
// soft-core mode the Coulomb term is K*q_i*q_j*l^p/sqrt(r^2+alpha*(1-l)^s);
// its derivative must change alongside the slot energy. V2 PME is handled by
// the caller: it deliberately does not use UV-scaled charges and has no delta.
template<class Radius>
__device__ __forceinline__ double uv_pair_charge_work(
    Radius r2, float product_before, float product_after, float scale_qq,
    bool lambda_pair, float lambda, float alpha_q_a2, float softcore_power,
    float coupling_power, double& delta_dudl
) {
    delta_dudl = 0.0;
    if (product_before == product_after) return 0.0;
    if (!lambda_pair) {
        const double inv_r = (double)bp_native_rsqrt(r2);
        const double before = (double)COULOMB_CONSTANT * (double)product_before * inv_r;
        const double after = (double)COULOMB_CONSTANT * (double)product_after * inv_r;
        return (double)scale_qq * (after - before);
    }
    const double l = (double)lambda;
    const double one_minus = 1.0 - l;
    const double scale = pow(l, (double)coupling_power);
    const double dscale = l == 0.0
        ? ((double)coupling_power == 1.0 ? 1.0 : 0.0)
        : (double)coupling_power * pow(l, (double)coupling_power - 1.0);
    const double soften = pow(one_minus, (double)softcore_power);
    const double dsoften = one_minus == 0.0
        ? ((double)softcore_power == 1.0 ? -1.0 : 0.0)
        : -(double)softcore_power * pow(one_minus, (double)softcore_power - 1.0);
    const double inv_sc = 1.0 / sqrt((double)r2 + (double)alpha_q_a2 * soften);
    const double dinv_sc = -0.5 * (double)alpha_q_a2 * dsoften * inv_sc*inv_sc*inv_sc;
    const double before = (double)COULOMB_CONSTANT * (double)product_before;
    const double after = (double)COULOMB_CONSTANT * (double)product_after;
    const double derivative_shape = dscale*inv_sc + scale*dinv_sc;
    delta_dudl = after*derivative_shape - before*derivative_shape;
    // V1 soft-core's existing Coulomb expression does not apply legacy FUDGE.
    return after*scale*inv_sc - before*scale*inv_sc;
}

// Fixed-coordinate UV parameter work, one writer per atom, no force or work
// atomics. Reuse the exact ordinary-neighbor and exception pair domains from
// deterministic force gather; the neighbor CSR already carries exclusions.
// Adding the same delta to PE and work puts the measured Hamiltonian and its
// external-work correction on the POST-pump charge-state boundary.
template<bool Compensated>
__device__ __forceinline__ void bp_native_uv_parameter_work_body(
    const float3* positions, const float* charges, const float* ground_charges,
    const int* atom_to_aromatic, const int* aromatic_type, const float* charge_set_mean,
    const int* excited_before, const float* population_before,
    const int* excited_after, const float* population_after,
    const int* neighbor_list, const int* n_neighbors,
    const int* pairs14_list, const int* pairs14_offsets, const float* pairs14_charge_product,
    int exact_exceptions, int n_atoms, float cutoff,
    const int* lambda_atom_slot, const float* lambda_slot_values, int n_lambda_slots,
    float alpha_q_a2, float softcore_power, float coupling_power, int lambda_v2_ewald_enabled,
    double* pe_components, double* external_work,
    double* lambda_dudl_atom, double* lambda_energy_atom
,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault
) {
    using G=BpNativeGeometry<Compensated>;using GVec=typename G::Vec;using GReal=typename G::Real;
    if constexpr(Compensated) {
        if(!bp_native_coordinate_owner_valid(coordinate_owner,reinterpret_cast<const float*>(positions),
            n_atoms,BP_COORD_CURRENT_INITIALIZED,precision_fault))return;
    }

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_atoms || lambda_v2_ewald_enabled != 0) return;
    float my_scale_before, my_scale_after;
    const float qi_before = uv_charge_for_state(tid, charges, ground_charges,
        atom_to_aromatic, aromatic_type, exact_exceptions < 0 ? nullptr : charge_set_mean,
        excited_before, population_before, my_scale_before);
    const float qi_after = uv_charge_for_state(tid, charges, ground_charges,
        atom_to_aromatic, aromatic_type, exact_exceptions < 0 ? nullptr : charge_set_mean,
        excited_after, population_after, my_scale_after);
    const int my_slot = lambda_atom_slot != nullptr ? lambda_atom_slot[tid] : -1;
    if (my_slot >= n_lambda_slots) { PRISM_PHYSICS_TRAP(); return; }
    const float li = my_slot >= 0 ? lambda_slot_values[my_slot] : 1.0f;
    const int start14 = pairs14_offsets[tid], end14 = pairs14_offsets[tid+1];
    const GVec pi = G::load(positions,tid,coordinate_owner,precision_fault);
    const float cutoff_sq = cutoff*cutoff;
    double work = 0.0, slot_energy = 0.0, slot_dudl = 0.0;
    // Pass 0 mirrors the ordinary neighbor loop, including legacy 1-4 FUDGE.
    // Pass 1 visits exact exceptions without the ordinary nonbonded cutoff.
    for (int pass = 0; pass < (exact_exceptions > 0 ? 2 : 1); ++pass) {
        const int count = pass == 0 ? n_neighbors[tid] : end14-start14;
        for (int k = 0; k < count; ++k) {
            const int j = pass == 0 ? neighbor_list[tid*NEIGHBOR_LIST_SIZE+k] : pairs14_list[start14+k];
            int pair14 = pass == 1 ? start14+k : -1;
            if (pass == 0) {
                for (int p = start14; p < end14; ++p) {
                    if (pairs14_list[p] == j) { pair14 = p; break; }
                }
                if (exact_exceptions > 0 && pair14 >= 0) continue;
            }
            const GVec pj = G::load(positions,j,coordinate_owner,precision_fault);
            const GReal dx = pi.x-pj.x, dy = pi.y-pj.y, dz = pi.z-pj.z;
            const GReal r2 = dx*dx + dy*dy + dz*dz;
            if constexpr(Compensated) {if(!isfinite(r2)){bp_native_coordinate_fault(precision_fault);return;}}
            if (pass == 0 && (r2 >= cutoff_sq || r2 < 0.01f)) continue;
            if (pass == 1 && r2 < 1.0e-8f) { PRISM_PHYSICS_TRAP(); return; }
            float other_scale_before, other_scale_after;
            const float qj_before = uv_charge_for_state(j, charges, ground_charges,
                atom_to_aromatic, aromatic_type, exact_exceptions < 0 ? nullptr : charge_set_mean,
                excited_before, population_before, other_scale_before);
            const float qj_after = uv_charge_for_state(j, charges, ground_charges,
                atom_to_aromatic, aromatic_type, exact_exceptions < 0 ? nullptr : charge_set_mean,
                excited_after, population_after, other_scale_after);
            // Preserve float products and exception scaling order used by
            // force gather before promoting the Coulomb energy to double.
            const float product_before = pass == 1
                ? pairs14_charge_product[pair14]*my_scale_before*other_scale_before
                : qi_before*qj_before;
            const float product_after = pass == 1
                ? pairs14_charge_product[pair14]*my_scale_after*other_scale_after
                : qi_after*qj_after;
            const int other_slot = lambda_atom_slot != nullptr ? lambda_atom_slot[j] : -1;
            if (other_slot >= n_lambda_slots) { PRISM_PHYSICS_TRAP(); return; }
            const float lj = other_slot >= 0 ? lambda_slot_values[other_slot] : 1.0f;
            const bool lambda_pair = my_slot >= 0 || other_slot >= 0;
            const float scale_qq = pass == 0 && pair14 >= 0 ? FUDGE_QQ : 1.0f;
            double delta_dudl = 0.0;
            double delta = uv_pair_charge_work(r2, product_before, product_after,
                scale_qq, lambda_pair, li*lj, alpha_q_a2, softcore_power, coupling_power, delta_dudl);
            if (pass == 0 && !lambda_pair && exact_exceptions >= 0) {
                // The cutoff reference depends on charge state too. Its exact
                // parameter change belongs in BOTH PE and UV work. Reuse the
                // same native Coulomb work arithmetic at the reference radius.
                double reference_dudl = 0.0;
                delta -= uv_pair_charge_work((GReal)cutoff_sq, product_before, product_after,
                    scale_qq, false, li*lj, alpha_q_a2, softcore_power, coupling_power, reference_dudl);
            }
            work += 0.5*delta;
            if (my_slot >= 0) {
                slot_energy += delta * (other_slot >= 0 ? 0.5 : 1.0);
                slot_dudl += delta_dudl * (double)lj;
            }
        }
    }
    if (!isfinite(work) || !isfinite(slot_energy) || !isfinite(slot_dudl)) {
        PRISM_PHYSICS_TRAP(); return;
    }
    if constexpr(Compensated) {if(atomicAdd(precision_fault,0u)!=0u)return;}
    pe_components[tid] += work;
    external_work[tid] += work;
    if (my_slot >= 0) {
        if (lambda_energy_atom != nullptr) lambda_energy_atom[tid] += slot_energy;
        if (lambda_dudl_atom != nullptr) lambda_dudl_atom[tid] += slot_dudl;
    }
}

extern "C" __global__ void nhs_uv_charge_parameter_work(
    const float3* positions, const float* charges, const float* ground_charges,
    const int* atom_to_aromatic, const int* aromatic_type, const float* charge_set_mean,
    const int* excited_before, const float* population_before,
    const int* excited_after, const float* population_after,
    const int* neighbor_list, const int* n_neighbors,
    const int* pairs14_list, const int* pairs14_offsets, const float* pairs14_charge_product,
    int exact_exceptions, int n_atoms, float cutoff,
    const int* lambda_atom_slot, const float* lambda_slot_values, int n_lambda_slots,
    float alpha_q_a2, float softcore_power, float coupling_power, int lambda_v2_ewald_enabled,
    double* pe_components, double* external_work,
    double* lambda_dudl_atom, double* lambda_energy_atom
) {
    bp_native_uv_parameter_work_body<false>(positions, charges, ground_charges, atom_to_aromatic, aromatic_type, charge_set_mean, excited_before, population_before, excited_after, population_after, neighbor_list, n_neighbors, pairs14_list, pairs14_offsets, pairs14_charge_product, exact_exceptions, n_atoms, cutoff, lambda_atom_slot, lambda_slot_values, n_lambda_slots, alpha_q_a2, softcore_power, coupling_power, lambda_v2_ewald_enabled, pe_components, external_work, lambda_dudl_atom, lambda_energy_atom, nullptr, nullptr);
}

extern "C" __global__ void nhs_uv_charge_parameter_work_compensated_v1(
    const float3* positions, const float* charges, const float* ground_charges,
    const int* atom_to_aromatic, const int* aromatic_type, const float* charge_set_mean,
    const int* excited_before, const float* population_before,
    const int* excited_after, const float* population_after,
    const int* neighbor_list, const int* n_neighbors,
    const int* pairs14_list, const int* pairs14_offsets, const float* pairs14_charge_product,
    int exact_exceptions, int n_atoms, float cutoff,
    const int* lambda_atom_slot, const float* lambda_slot_values, int n_lambda_slots,
    float alpha_q_a2, float softcore_power, float coupling_power, int lambda_v2_ewald_enabled,
    double* pe_components, double* external_work,
    double* lambda_dudl_atom, double* lambda_energy_atom
,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault
) {
    bp_native_uv_parameter_work_body<true>(positions, charges, ground_charges, atom_to_aromatic, aromatic_type, charge_set_mean, excited_before, population_before, excited_after, population_after, neighbor_list, n_neighbors, pairs14_list, pairs14_offsets, pairs14_charge_product, exact_exceptions, n_atoms, cutoff, lambda_atom_slot, lambda_slot_values, n_lambda_slots, alpha_q_a2, softcore_power, coupling_power, lambda_v2_ewald_enabled, pe_components, external_work, lambda_dudl_atom, lambda_energy_atom, coordinate_owner, precision_fault);
}

extern "C" __global__ void nhs_uv_pump_probe_step(
    ProtocolState* d_protocol,
    float3* velocities,
    const float* masses,
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    const int* d_aromatic_type,
    const float3* d_ring_normals,
    const AromaticNeighbors* d_aromatic_neighbors,
    int n_aromatics,
    double* d_external_work
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (n_aromatics <= 0) return;

    const int uv_burst_active = d_protocol->uv_burst_active;
    const float uv_wavelength_nm = d_protocol->uv_wavelength_nm;
    const float dt = d_protocol->dt;
    const int timestep = (int)d_protocol->current_step;

#ifdef DEBUG_UV_WAVELENGTH
    printf("[GPU] uv_wavelength_nm=%.2f uv_burst_active=%d n_aromatics=%d timestep=%d\n",
           uv_wavelength_nm, uv_burst_active, n_aromatics, timestep);
#endif

    if (uv_burst_active) {
        const float uv_burst_energy_override = d_protocol->uv_burst_energy;
        for (int arom_idx = 0; arom_idx < n_aromatics; arom_idx++) {
            if (d_is_excited[arom_idx] == 0) {
                excite_aromatic_wavelength(
                    arom_idx,
                    d_aromatic_type[arom_idx],
                    uv_wavelength_nm,
                    d_is_excited,
                    d_time_since_excitation,
                    d_electronic_population,
                    d_vibrational_energy,
                    d_franck_condon_progress,
                    uv_burst_energy_override
                );
            }
        }
    }

    for (int arom_idx = 0; arom_idx < n_aromatics; arom_idx++) {
        float energy_to_transfer = 0.0f;
        update_excited_state_inline(
            arom_idx,
            d_aromatic_type[arom_idx],
            dt,
            d_is_excited,
            d_time_since_excitation,
            d_electronic_population,
            d_vibrational_energy,
            d_franck_condon_progress,
            &energy_to_transfer
        );

        if (energy_to_transfer > 0.001f && d_aromatic_neighbors != nullptr) {
            AromaticNeighbors neighbors = d_aromatic_neighbors[arom_idx];
            apply_vibrational_transfer(
                arom_idx,
                energy_to_transfer,
                d_ring_normals[arom_idx],
                velocities,
                masses,
                neighbors.atom_indices,
                neighbors.n_neighbors,
                timestep * n_aromatics + arom_idx,
                d_aromatic_type[arom_idx],
                uv_wavelength_nm,
                d_external_work
            );
        }
    }
}

// P3 — W_ext window reduction.
//
// PRIOR DEFECT: this kernel summed the per-atom plane 0..n_atoms-1 and then
// wrote the total back into `d_external_work[0]` — the SAME cell that is
// atom 0's own per-atom slot. Combined with the MD's per-step full-plane
// memset, the scalar could never hold more than ONE step's work, so the
// "integrated over this chunk window" semantics the ZSTR schema and the
// gearbox SFA both depend on were unsatisfiable by the data layout, quite
// independently of any memset placed elsewhere.
//
// FIX: the reduction sink is now a SEPARATE single-element cell that no atom
// ever writes and the per-step plane memset never touches. The reducer
// ACCUMULATES into it (atomicAdd, not assignment), so across a replay the cell
// holds the true window integral. The adjudication graph zeroes that cell at
// replay head, which closes the window exactly where `cruise->v_prev` latches —
// so the W_ext window and the v_prev window finally describe the same interval.
//
// `d_wext_window` may be null on legacy call paths; the reduction then degrades
// to a no-op rather than writing a plausible value into the atom plane. The SFA
// reads UNMEASURED in that case instead of a fabricated zero.
extern "C" __global__ void nhs_reduce_external_work_components(
    double* d_external_work,
    int n_atoms,
    double* d_wext_window
) {
    __shared__ double partial[256];
    const int lane = threadIdx.x;
    double sum = 0.0;

    if (d_external_work != nullptr) {
        for (int i = lane; i < n_atoms; i += blockDim.x) {
            sum += d_external_work[i];
        }
    }
    partial[lane] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0 && d_external_work != nullptr && d_wext_window != nullptr) {
        // Accumulate, never assign: the window spans a whole replay, while this
        // kernel runs once per step. Atom 0's slot is left untouched.
        atomicAdd(d_wext_window, partial[0]);
    }
}

extern "C" __global__ void nhs_rattle_velocity_constraints(
    const float3* positions,
    float3* velocities,
    const HCluster* h_clusters,
    int n_clusters,
    double* d_nonham_work,
    PrismConstraintWindowV1* d_constraint_window
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int c = tid; c < n_clusters; c += gridDim.x * blockDim.x) {
        if (!rattle_cluster_velocities(
                positions, velocities, h_clusters[c], d_nonham_work, 100)) {
            prism_constraint_failure(d_constraint_window);
                PRISM_PHYSICS_TRAP();
            return;
        }
        prism_constraint_success(d_constraint_window, 2, c);
    }
}

extern "C" __global__ void nhs_velocity_safety_clamp(
    float3* velocities,
    const float* masses,
    int n_atoms,
    const ProtocolState* d_protocol,
    double* d_nonham_work
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_atoms) return;

    const float mass = masses[tid];
    const float target_temp = d_protocol->current_temperature;
    const float thermal_velocity = sqrtf(
        AMBER_KCAL_TO_AMU_A2_PS2 * KB_KCAL_MOL_K * target_temp / fmaxf(mass, 1.0f));
    const float max_velocity = fmaxf(20.0f * thermal_velocity, 2.0f);
    const float3 old_velocity = velocities[tid];
    const float magnitude = sqrtf(old_velocity.x * old_velocity.x
                                 + old_velocity.y * old_velocity.y
                                 + old_velocity.z * old_velocity.z);
    if (magnitude <= max_velocity) return;

    const float scale = max_velocity / magnitude;
    const float3 new_velocity = make_float3(
        old_velocity.x * scale,
        old_velocity.y * scale,
        old_velocity.z * scale);
    velocities[tid] = new_velocity;
    if (d_nonham_work != nullptr) {
        const double old_sq = (double)old_velocity.x * old_velocity.x
                            + (double)old_velocity.y * old_velocity.y
                            + (double)old_velocity.z * old_velocity.z;
        const double new_sq = (double)new_velocity.x * new_velocity.x
                            + (double)new_velocity.y * new_velocity.y
                            + (double)new_velocity.z * new_velocity.z;
        d_nonham_work[tid] += 0.5 * (double)mass * (new_sq - old_sq)
                            / (double)AMBER_KCAL_TO_AMU_A2_PS2;
    }
}

// Explicit selected-profile endpoint owners; legacy endpoints remain unchanged.
extern "C" __global__ void nhs_biprism_rattle_velocity_constraints_v1(
    const float3* positions,
    float3* velocities,
    const HCluster* h_clusters,
    int n_clusters,
    double* d_nonham_work,
    PrismConstraintWindowV1* d_constraint_window,
    const uint8_t* d_biprism_ownership,
    int n_atoms
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (!d_biprism_ownership) { PRISM_PHYSICS_TRAP(); return; }
    for (int c = tid; c < n_clusters; c += gridDim.x * blockDim.x) {
        const int action = prism_biprism_constraint_action_v1(d_biprism_ownership, n_atoms, h_clusters[c]);
        if (action < 0) { prism_constraint_failure(d_constraint_window); PRISM_PHYSICS_TRAP(); return; }
        if (action > 0) continue; // No false success stamp for a flexible component.
        if (!rattle_cluster_velocities(
                positions, velocities, h_clusters[c], d_nonham_work, 100)) {
            prism_constraint_failure(d_constraint_window);
                PRISM_PHYSICS_TRAP();
            return;
        }
        prism_constraint_success(d_constraint_window, 2, c);
    }
}

extern "C" __global__ void nhs_biprism_rattle_velocity_constraints_compensated_v1(
    const float3* positions,
    float3* velocities,
    const HCluster* h_clusters,
    int n_clusters,
    double* d_nonham_work,
    PrismConstraintWindowV1* d_constraint_window,
    const uint8_t* d_biprism_ownership,
    int n_atoms,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(!bp_native_coordinate_owner_valid(coordinate_owner,reinterpret_cast<const float*>(positions),
        n_atoms,BP_COORD_CURRENT_INITIALIZED,precision_fault))return;
    if (!d_biprism_ownership) { PRISM_PHYSICS_TRAP(); return; }
    for (int c = tid; c < n_clusters; c += gridDim.x * blockDim.x) {
        const int action = prism_biprism_constraint_action_v1(d_biprism_ownership, n_atoms, h_clusters[c]);
        if (action < 0) { prism_constraint_failure(d_constraint_window); PRISM_PHYSICS_TRAP(); return; }
        if (action > 0) continue; // No false success stamp for a flexible component.
        if (!rattle_cluster_velocities<true>(
                positions, velocities, h_clusters[c], d_nonham_work, 100,coordinate_owner,precision_fault)) {
            prism_constraint_failure(d_constraint_window);
                PRISM_PHYSICS_TRAP();
            return;
        }
        prism_constraint_success(d_constraint_window, 2, c);
    }
}

extern "C" __global__ void nhs_biprism_velocity_safety_clamp_v1(
    float3* velocities,
    const float* masses,
    int n_atoms,
    const ProtocolState* d_protocol,
    double* d_nonham_work,
    const uint8_t* d_biprism_ownership
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_atoms) return;
    if (!d_biprism_ownership || !prism_biprism_ownership_valid_v1(d_biprism_ownership[tid])) {
        PRISM_PHYSICS_TRAP(); return;
    }
    if (prism_biprism_external_motion_v1(d_biprism_ownership[tid])) return;


    const float mass = masses[tid];
    const float target_temp = d_protocol->current_temperature;
    const float thermal_velocity = sqrtf(
        AMBER_KCAL_TO_AMU_A2_PS2 * KB_KCAL_MOL_K * target_temp / fmaxf(mass, 1.0f));
    const float max_velocity = fmaxf(20.0f * thermal_velocity, 2.0f);
    const float3 old_velocity = velocities[tid];
    const float magnitude = sqrtf(old_velocity.x * old_velocity.x
                                 + old_velocity.y * old_velocity.y
                                 + old_velocity.z * old_velocity.z);
    if (magnitude <= max_velocity) return;

    const float scale = max_velocity / magnitude;
    const float3 new_velocity = make_float3(
        old_velocity.x * scale,
        old_velocity.y * scale,
        old_velocity.z * scale);
    velocities[tid] = new_velocity;
    if (d_nonham_work != nullptr) {
        const double old_sq = (double)old_velocity.x * old_velocity.x
                            + (double)old_velocity.y * old_velocity.y
                            + (double)old_velocity.z * old_velocity.z;
        const double new_sq = (double)new_velocity.x * new_velocity.x
                            + (double)new_velocity.y * new_velocity.y
                            + (double)new_velocity.z * new_velocity.z;
        d_nonham_work[tid] += 0.5 * (double)mass * (new_sq - old_sq)
                            / (double)AMBER_KCAL_TO_AMU_A2_PS2;
    }
}

// ============================================================================
// INITIALIZATION KERNELS
// ============================================================================

// Initialize EFP state
extern "C" __global__ void init_efp_state(
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int total_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_voxels) {
        efp_potential[idx] = 0.0f;
        efp_potential_prev[idx] = 0.0f;
        efp_lif_potential[idx] = 0.0f;
    }
}

extern "C" __global__ void init_rng_states(
    curandState* states,
    unsigned long long seed,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Standalone mutable initializer only. The native engine's published shared
// spatial bank is constructor-built and must NEVER be passed to this writer.
extern "C" __global__ void init_warp_matrix(
    WarpEntry* warp_matrix,
    const float3* positions,
    int n_atoms,
    float3 grid_origin,
    float grid_spacing,
    int grid_dim,
    float cutoff
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = grid_dim * grid_dim * grid_dim;

    for (int v = tid; v < total_voxels; v += gridDim.x * blockDim.x) {
        int vz = v / (grid_dim * grid_dim);
        int vy = (v / grid_dim) % grid_dim;
        int vx = v % grid_dim;

        float3 voxel_center = make_float3(
            grid_origin.x + (vx + 0.5f) * grid_spacing,
            grid_origin.y + (vy + 0.5f) * grid_spacing,
            grid_origin.z + (vz + 0.5f) * grid_spacing
        );

        warp_matrix[v].voxel_idx = v;
        build_warp_entry(warp_matrix[v], voxel_center, positions, n_atoms, cutoff);
    }
}

extern "C" __global__ void init_lif_state(
    float* lif_potential,
    float* water_density,
    float* water_density_prev,
    int* spike_grid,
    int total_voxels
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int v = tid; v < total_voxels; v += gridDim.x * blockDim.x) {
        lif_potential[v] = 0.0f;
        water_density[v] = WATER_DENSITY_BULK;
        water_density_prev[v] = WATER_DENSITY_BULK;
        spike_grid[v] = 0;
    }
}

// ============================================================================
// TEMPERATURE PROTOCOL HELPERS
// ============================================================================

extern "C" __global__ void set_temperature_protocol(
    TemperatureProtocol* protocol,
    float start_temp,
    float end_temp,
    int ramp_steps,
    int hold_steps
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        protocol->start_temp = start_temp;
        protocol->end_temp = end_temp;
        protocol->ramp_steps = ramp_steps;
        protocol->hold_steps = hold_steps;
        protocol->current_step = 0;
    }
}

extern "C" __global__ void advance_temperature_protocol(
    TemperatureProtocol* protocol
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        protocol->current_step++;
    }
}

// ============================================================================
// EXCITED STATE INITIALIZATION
// ============================================================================

extern "C" __global__ void init_excited_state(
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    int n_aromatics
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_aromatics) {
        d_is_excited[tid] = 0;
        d_time_since_excitation[tid] = 0.0f;
        d_electronic_population[tid] = 0.0f;
        d_vibrational_energy[tid] = 0.0f;
        d_franck_condon_progress[tid] = 0.0f;
    }
}

extern "C" __global__ void init_atom_to_aromatic(
    int* d_atom_to_aromatic,
    int n_atoms
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_atoms) {
        d_atom_to_aromatic[tid] = -1;  // Not an aromatic atom by default
    }
}

extern "C" __global__ void build_aromatic_neighbors(
    AromaticNeighbors* d_aromatic_neighbors,
    const float3* positions,
    const int* aromatic_atom_indices,  // [n_aromatics * MAX_RING_ATOMS]
    const int* aromatic_n_atoms,       // [n_aromatics]
    int n_aromatics,
    int n_atoms,
    float neighbor_cutoff              // ~5 Angstroms
) {
    int arom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (arom_idx < n_aromatics) {
        AromaticNeighbors* neighbors = &d_aromatic_neighbors[arom_idx];
        neighbors->n_neighbors = 0;

        // Get center of aromatic ring
        float3 ring_center = make_float3(0, 0, 0);
        int n_ring_atoms = aromatic_n_atoms[arom_idx];

        for (int i = 0; i < n_ring_atoms && i < 16; i++) {
            int atom_idx = aromatic_atom_indices[arom_idx * 16 + i];
            if (atom_idx >= 0 && atom_idx < n_atoms) {
                ring_center.x += positions[atom_idx].x;
                ring_center.y += positions[atom_idx].y;
                ring_center.z += positions[atom_idx].z;
            }
        }
        if (n_ring_atoms > 0) {
            ring_center.x /= n_ring_atoms;
            ring_center.y /= n_ring_atoms;
            ring_center.z /= n_ring_atoms;
        }

        float cutoff_sq = neighbor_cutoff * neighbor_cutoff;

        // Find all atoms within cutoff of ring center
        for (int a = 0; a < n_atoms && neighbors->n_neighbors < 64; a++) {
            // Skip atoms that are part of this aromatic ring
            bool is_ring_atom = false;
            for (int i = 0; i < n_ring_atoms && i < 16; i++) {
                if (aromatic_atom_indices[arom_idx * 16 + i] == a) {
                    is_ring_atom = true;
                    break;
                }
            }
            if (is_ring_atom) continue;

            float dx = positions[a].x - ring_center.x;
            float dy = positions[a].y - ring_center.y;
            float dz = positions[a].z - ring_center.z;
            float dist_sq = dx*dx + dy*dy + dz*dz;

            if (dist_sq < cutoff_sq) {
                neighbors->atom_indices[neighbors->n_neighbors] = a;
                neighbors->n_neighbors++;
            }
        }
    }
}

extern "C" __global__ void compute_ring_normals(
    float3* d_ring_normals,
    const float3* positions,
    const int* aromatic_atom_indices,  // [n_aromatics * 16]
    const int* aromatic_n_atoms,       // [n_aromatics]
    int n_aromatics,
    int n_atoms
) {
    int arom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (arom_idx < n_aromatics) {
        int n_ring_atoms = aromatic_n_atoms[arom_idx];

        // Need at least 3 atoms to compute a plane normal
        if (n_ring_atoms < 3) {
            d_ring_normals[arom_idx] = make_float3(0, 0, 1);  // Default
            return;
        }

        // P5 Step 6 — COVARIANCE BEST-FIT NORMAL over the WHOLE ring.
        //
        // The prior implementation took the cross product of the first three
        // entries only. On an exactly planar ring that is correct, but real
        // rings pucker, and three points cannot see it: measured against a
        // covariance best fit over this topology's 30 aromatics, the three-point
        // estimate carries a mean 1.56 deg (max 8.14 deg) error even when it is
        // handed the CORRECT ring atoms. It is also maximally sensitive to which
        // three atoms happen to sit first in the list.
        //
        // The normal is the eigenvector of the SMALLEST eigenvalue of the atom
        // covariance about the ring centroid. Rather than solve the cubic, note
        // that for symmetric C the matrix (trace(C)*I - C) has that same vector
        // as its LARGEST eigenvector, so plain power iteration converges to it —
        // and converges fast, because a near-planar ring has one eigenvalue
        // orders of magnitude below the other two. Order-independent by
        // construction: no assumption that the ring atoms are listed in ring
        // order, which they are not.
        float cx = 0.0f, cy = 0.0f, cz = 0.0f;
        int n_used = 0;
        for (int k = 0; k < n_ring_atoms && k < 16; ++k) {
            int a = aromatic_atom_indices[arom_idx * 16 + k];
            if (a < 0 || a >= n_atoms) continue;
            cx += positions[a].x; cy += positions[a].y; cz += positions[a].z;
            ++n_used;
        }
        if (n_used < 3) {
            d_ring_normals[arom_idx] = make_float3(0, 0, 1);
            return;
        }
        const float inv_n = 1.0f / (float)n_used;
        cx *= inv_n; cy *= inv_n; cz *= inv_n;

        float cxx = 0.0f, cxy = 0.0f, cxz = 0.0f, cyy = 0.0f, cyz = 0.0f, czz = 0.0f;
        for (int k = 0; k < n_ring_atoms && k < 16; ++k) {
            int a = aromatic_atom_indices[arom_idx * 16 + k];
            if (a < 0 || a >= n_atoms) continue;
            const float dx = positions[a].x - cx;
            const float dy = positions[a].y - cy;
            const float dz = positions[a].z - cz;
            cxx += dx*dx; cxy += dx*dy; cxz += dx*dz;
            cyy += dy*dy; cyz += dy*dz; czz += dz*dz;
        }

        // M = trace(C) * I - C  (symmetric; largest eigenvector of M == smallest of C)
        const float tr = cxx + cyy + czz;
        const float m00 = tr - cxx, m01 = -cxy,      m02 = -cxz;
        const float m11 = tr - cyy, m12 = -cyz,      m22 = tr - czz;

        // Seeded off-axis so the iterate is not accidentally orthogonal to the
        // target eigenvector for an axis-aligned ring.
        float vx = 0.5773502692f, vy = 0.5773502692f, vz = 0.5773502692f;
        for (int it = 0; it < 48; ++it) {
            const float nx = m00*vx + m01*vy + m02*vz;
            const float ny = m01*vx + m11*vy + m12*vz;
            const float nz = m02*vx + m12*vy + m22*vz;
            const float ln = sqrtf(nx*nx + ny*ny + nz*nz);
            if (ln <= 1e-20f) break;
            const float inv = 1.0f / ln;
            vx = nx * inv; vy = ny * inv; vz = nz * inv;
        }
        float3 normal = make_float3(vx, vy, vz);

        // Normalize
        float len = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
        if (len > 1e-8f) {
            normal.x /= len;
            normal.y /= len;
            normal.z /= len;
        } else {
            normal = make_float3(0, 0, 1);
        }

        d_ring_normals[arom_idx] = normal;
    }
}

// The compensated profile refreshes the same two derived float3 planes from
// current native coordinates. Ring membership remains the topology authority.
// The covariance/power-iteration normal is the existing compute_ring_normals
// equation and iteration count; only its geometric arithmetic is widened.
__device__ __forceinline__ bool bp_native_aromatic_geometry_candidate(
    const BiprismCoordinateOwnerFfi& owner,const int* indices,const int* counts,
    int ring,int atoms,float3& centroid,float3& normal
) {
    const int count=counts[ring];
    if(count<3 || count>16)return false;
    double cx=0.,cy=0.,cz=0.;
    for(int k=0;k<count;++k) {
        const int a=indices[ring*16+k];
        if(a<0 || a>=atoms)return false;
        for(int j=0;j<k;++j)if(indices[ring*16+j]==a)return false;
        const auto pair=bp_coord_load(owner,BiprismCoordinatePlane::Current,(uint32_t)a);
        if(!bp_coord_finite(pair))return false;
        const auto x=bp_coord_value(pair);
        cx+=x.x;cy+=x.y;cz+=x.z;
    }
    const double inv_n=1./(double)count;
    cx*=inv_n;cy*=inv_n;cz*=inv_n;
    if(!isfinite(cx)||!isfinite(cy)||!isfinite(cz))return false;
    double cxx=0.,cxy=0.,cxz=0.,cyy=0.,cyz=0.,czz=0.;
    for(int k=0;k<count;++k) {
        const auto x=bp_coord_value(bp_coord_load(owner,BiprismCoordinatePlane::Current,
            (uint32_t)indices[ring*16+k]));
        const double dx=x.x-cx,dy=x.y-cy,dz=x.z-cz;
        cxx+=dx*dx;cxy+=dx*dy;cxz+=dx*dz;
        cyy+=dy*dy;cyz+=dy*dz;czz+=dz*dz;
    }
    const double tr=cxx+cyy+czz;
    const double m00=tr-cxx,m01=-cxy,m02=-cxz,m11=tr-cyy,m12=-cyz,m22=tr-czz;
    if(!isfinite(m00)||!isfinite(m01)||!isfinite(m02)||
       !isfinite(m11)||!isfinite(m12)||!isfinite(m22))return false;
    double vx=(double)0.5773502692f,vy=vx,vz=vx;
    for(int it=0;it<48;++it) {
        const double nx=m00*vx+m01*vy+m02*vz;
        const double ny=m01*vx+m11*vy+m12*vz;
        const double nz=m02*vx+m12*vy+m22*vz;
        const double length=sqrt(nx*nx+ny*ny+nz*nz);
        if(!isfinite(length))return false;
        if(length<=(double)1e-20f)break;
        const double inv=1./length;vx=nx*inv;vy=ny*inv;vz=nz*inv;
    }
    const double length=sqrt(vx*vx+vy*vy+vz*vz);
    if(!isfinite(length))return false;
    if(length>(double)1e-8f){vx/=length;vy/=length;vz/=length;}
    else {vx=0.;vy=0.;vz=1.;}
    centroid=make_float3((float)cx,(float)cy,(float)cz);
    normal=make_float3((float)vx,(float)vy,(float)vz);
    return isfinite(centroid.x)&&isfinite(centroid.y)&&isfinite(centroid.z)&&
        isfinite(normal.x)&&isfinite(normal.y)&&isfinite(normal.z);
}

extern "C" __global__ void __launch_bounds__(256) nhs_refresh_aromatic_geometry_compensated_v1(
    const float3* positions,const int* aromatic_atom_indices,const int* aromatic_n_atoms,
    int n_aromatics,int n_atoms,float3* centroids,float3* ring_normals,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault
) {
    if(gridDim.x!=1 || gridDim.y!=1 || gridDim.z!=1 ||
       blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1) {
        if(threadIdx.x==0)bp_native_coordinate_fault(precision_fault);return;
    }
    const bool bound=bp_native_coordinate_owner_valid(coordinate_owner,
        reinterpret_cast<const float*>(positions),n_atoms,BP_COORD_CURRENT_INITIALIZED,precision_fault);
    const bool shape=n_aromatics>=0 && n_aromatics<=n_atoms &&
        (n_aromatics==0 || (aromatic_atom_indices && aromatic_n_atoms && centroids && ring_normals &&
            centroids!=ring_normals && centroids!=positions && ring_normals!=positions));
    if(!__syncthreads_and(bound&&shape)) {
        if(threadIdx.x==0)bp_native_coordinate_fault(precision_fault);return;
    }
    bool valid=true;
    for(int ring=(int)threadIdx.x;ring<n_aromatics;ring+=(int)blockDim.x) {
        float3 center,normal;
        if(!bp_native_aromatic_geometry_candidate(*coordinate_owner,aromatic_atom_indices,
            aromatic_n_atoms,ring,n_atoms,center,normal))valid=false;
    }
    // Validate every ring before publishing either plane. The same-stream
    // owner freezes coordinate/membership inputs throughout this launch;
    // recomputation avoids an artificial maximum on the ring population.
    if(!__syncthreads_and(valid&&atomicAdd(precision_fault,0u)==0u)) {
        if(threadIdx.x==0)bp_native_coordinate_fault(precision_fault);return;
    }
    for(int ring=(int)threadIdx.x;ring<n_aromatics;ring+=(int)blockDim.x) {
        float3 center,normal;
        if(!bp_native_aromatic_geometry_candidate(*coordinate_owner,aromatic_atom_indices,
            aromatic_n_atoms,ring,n_atoms,center,normal)) {
            bp_native_coordinate_fault(precision_fault);return;
        }
        centroids[ring]=center;ring_normals[ring]=normal;
    }
}

// ============================================================================
// O(N) CELL LIST CONSTRUCTION
// ============================================================================

/**
 * @brief Build cell lists from atom positions
 *
 * Each atom is assigned to exactly one cell based on its position.
 * Cell index = ix + iy * nx + iz * nx * ny
 *
 * Call this BEFORE build_neighbor_list, typically every 10-20 steps.
 */
extern "C" __global__ void build_cell_list(
    const float3* __restrict__ positions,  // [n_atoms]
    int* __restrict__ cell_list,           // [MAX_TOTAL_CELLS * MAX_ATOMS_PER_CELL]
    int* __restrict__ cell_counts,         // [MAX_TOTAL_CELLS]
    int* __restrict__ atom_cell,           // [n_atoms] - which cell each atom is in
    float origin_x, float origin_y, float origin_z,
    int nx, int ny, int nz,
    int n_atoms
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_atoms) return;

    float x = positions[tid].x;
    float y = positions[tid].y;
    float z = positions[tid].z;

    // Compute cell indices
    int ix = (int)((x - origin_x) * CELL_SIZE_INV);
    int iy = (int)((y - origin_y) * CELL_SIZE_INV);
    int iz = (int)((z - origin_z) * CELL_SIZE_INV);

    // Clamp to valid range
    ix = max(0, min(ix, nx - 1));
    iy = max(0, min(iy, ny - 1));
    iz = max(0, min(iz, nz - 1));

    int cell_idx = ix + iy * nx + iz * nx * ny;
    atom_cell[tid] = cell_idx;

    // Atomically add atom to cell
    int slot = atomicAdd(&cell_counts[cell_idx], 1);
    if (slot < MAX_ATOMS_PER_CELL) {
        cell_list[cell_idx * MAX_ATOMS_PER_CELL + slot] = tid;
    } else {
        // A truncated cell list corrupts the force field. Device-trap so a
        // captured/resident graph also fails closed without host readback.
        PRISM_PHYSICS_TRAP();
    }
}

extern "C" __global__ void nhs_build_cell_list_compensated_v1(
    const float3* __restrict__ positions,  // [n_atoms]
    int* __restrict__ cell_list,           // [MAX_TOTAL_CELLS * MAX_ATOMS_PER_CELL]
    int* __restrict__ cell_counts,         // [MAX_TOTAL_CELLS]
    int* __restrict__ atom_cell,           // [n_atoms] - which cell each atom is in
    float origin_x, float origin_y, float origin_z,
    int nx, int ny, int nz,
    int n_atoms,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(!bp_native_coordinate_owner_valid(coordinate_owner,reinterpret_cast<const float*>(positions),
        n_atoms,BP_COORD_CURRENT_INITIALIZED,precision_fault))return;
    if(nx<=0||ny<=0||nz<=0||!isfinite(origin_x)||!isfinite(origin_y)||!isfinite(origin_z)) {
        bp_native_coordinate_fault(precision_fault);return;
    }
    if (tid >= n_atoms) return;

    const auto point=BpNativeGeometry<true>::load(positions,tid,coordinate_owner,precision_fault);
    const double gx=(point.x-(double)origin_x)*(double)CELL_SIZE_INV;
    const double gy=(point.y-(double)origin_y)*(double)CELL_SIZE_INV;
    const double gz=(point.z-(double)origin_z)*(double)CELL_SIZE_INV;
    if(!isfinite(gx)||!isfinite(gy)||!isfinite(gz)) {bp_native_coordinate_fault(precision_fault);return;}
    // Clamp before the cast so even finite distant atoms cannot invoke an
    // out-of-range integer conversion. Interior truncation is the native rule.
    int ix=(int)fmax(0.0,fmin(gx,(double)(nx-1)));
    int iy=(int)fmax(0.0,fmin(gy,(double)(ny-1)));
    int iz=(int)fmax(0.0,fmin(gz,(double)(nz-1)));
    // Clamp to valid range
    ix = max(0, min(ix, nx - 1));
    iy = max(0, min(iy, ny - 1));
    iz = max(0, min(iz, nz - 1));

    int cell_idx = ix + iy * nx + iz * nx * ny;
    atom_cell[tid] = cell_idx;

    // Atomically add atom to cell
    int slot = atomicAdd(&cell_counts[cell_idx], 1);
    if (slot < MAX_ATOMS_PER_CELL) {
        cell_list[cell_idx * MAX_ATOMS_PER_CELL + slot] = tid;
    } else {
        // A truncated cell list corrupts the force field. Device-trap so a
        // captured/resident graph also fails closed without host readback.
        bp_native_coordinate_fault(precision_fault);
        PRISM_PHYSICS_TRAP();
    }
}

/**
 * @brief Reset cell counts to zero
 *
 * Call this before build_cell_list to clear previous frame's data.
 */
extern "C" __global__ void reset_cell_counts(
    int* __restrict__ cell_counts,
    int n_cells
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_cells) {
        cell_counts[tid] = 0;
    }
}

/**
 * @brief Build neighbor lists from cell lists (O(N) average case)
 *
 * For each atom, find all neighbors within cutoff by checking
 * the 27 cells (self + 26 neighbors). This replaces O(N²) all-pairs.
 *
 * Performance: For 3000 atoms with 10Å cutoff, typically ~100-200 neighbors/atom
 * instead of checking all 3000 pairs.
 */
extern "C" __global__ void build_neighbor_list(
    const float3* __restrict__ positions,  // [n_atoms]
    const int* __restrict__ cell_list,     // [MAX_TOTAL_CELLS * MAX_ATOMS_PER_CELL]
    const int* __restrict__ cell_counts,   // [MAX_TOTAL_CELLS]
    const int* __restrict__ atom_cell,     // [n_atoms]
    const int* __restrict__ excl_list,     // CSR exclusion list
    const int* __restrict__ excl_offsets,  // CSR offsets [n_atoms + 1]
    int* __restrict__ neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE]
    int* __restrict__ n_neighbors,         // [n_atoms]
    int nx, int ny, int nz,
    int n_atoms,
    float cutoff_sq                        // Squared cutoff with buffer
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_atoms) return;

    float3 my_pos = positions[tid];
    int my_cell = atom_cell[tid];
    int my_ix = my_cell % nx;
    int my_iy = (my_cell / nx) % ny;
    int my_iz = my_cell / (nx * ny);

    // Get my exclusion list
    int excl_start = excl_offsets[tid];
    int excl_end = excl_offsets[tid + 1];

    int neighbor_count = 0;
    int* my_neighbors = &neighbor_list[tid * NEIGHBOR_LIST_SIZE];

    // Check 27 neighboring cells (including self)
    for (int dz = -1; dz <= 1; dz++) {
        int iz = my_iz + dz;
        if (iz < 0 || iz >= nz) continue;

        for (int dy = -1; dy <= 1; dy++) {
            int iy = my_iy + dy;
            if (iy < 0 || iy >= ny) continue;

            for (int dx = -1; dx <= 1; dx++) {
                int ix = my_ix + dx;
                if (ix < 0 || ix >= nx) continue;

                int neighbor_cell = ix + iy * nx + iz * nx * ny;
                int n_in_cell = cell_counts[neighbor_cell];
                if (n_in_cell > MAX_ATOMS_PER_CELL) n_in_cell = MAX_ATOMS_PER_CELL;

                // Check all atoms in this cell
                for (int k = 0; k < n_in_cell; k++) {
                    int j = cell_list[neighbor_cell * MAX_ATOMS_PER_CELL + k];
                    if (j == tid) continue;

                    // Distance check
                    float dx_ij = positions[j].x - my_pos.x;
                    float dy_ij = positions[j].y - my_pos.y;
                    float dz_ij = positions[j].z - my_pos.z;
                    float r2 = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;

                    // Skip if outside cutoff (with buffer for list reuse)
                    if (r2 > cutoff_sq) continue;

                    // Check exclusion list (bonded pairs)
                    bool excluded = false;
                    for (int e = excl_start; e < excl_end; e++) {
                        if (excl_list[e] == j) {
                            excluded = true;
                            break;
                        }
                    }
                    if (excluded) continue;

                    // Add to neighbor list
                    if (neighbor_count < NEIGHBOR_LIST_SIZE) {
                        my_neighbors[neighbor_count] = j;
                        neighbor_count++;
                    } else {
                        // Never permit order-dependent neighbor truncation,
                        // including inside a metal-resident graph replay.
                        PRISM_PHYSICS_TRAP();
                    }
                }
            }
        }
    }

    // Canonical ascending atom IDs preserve the exact force summation order.
    // Heap sort uses bounded O(n log n) global-memory work instead of insertion
    // sort's quadratic row shifts; no neighbor, cutoff or rebuild is omitted.
    for (int start = neighbor_count / 2 - 1; start >= 0; --start) {
        int parent = start;
        const int key = my_neighbors[parent];
        for (int child = 2 * parent + 1; child < neighbor_count; child = 2 * parent + 1) {
            if (child + 1 < neighbor_count && my_neighbors[child] < my_neighbors[child + 1]) ++child;
            if (key >= my_neighbors[child]) break;
            my_neighbors[parent] = my_neighbors[child];
            parent = child;
        }
        my_neighbors[parent] = key;
    }
    for (int end = neighbor_count - 1; end > 0; --end) {
        const int key = my_neighbors[end];
        my_neighbors[end] = my_neighbors[0];
        int parent = 0;
        for (int child = 1; child < end; child = 2 * parent + 1) {
            if (child + 1 < end && my_neighbors[child] < my_neighbors[child + 1]) ++child;
            if (key >= my_neighbors[child]) break;
            my_neighbors[parent] = my_neighbors[child];
            parent = child;
        }
        my_neighbors[parent] = key;
    }

    n_neighbors[tid] = neighbor_count;
}

#include "nhs_neighbor_warp.cuh"

/**
 * @brief Compute nonbonded forces using neighbor lists (O(N))
 *
 * This is the fast path - uses precomputed neighbor lists instead of
 * O(N²) all-pairs. Should be ~50-100x faster for large proteins.
 *
 * NOTE: This kernel can be called instead of the inline nonbonded loop
 * in nhs_amber_fused_step for systems where neighbor list rebuild
 * overhead is worth the per-step savings.
 */
extern "C" __global__ void compute_nonbonded_neighborlist(
    const float3* __restrict__ positions,
    float3* __restrict__ forces,
    const float* __restrict__ charges,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ neighbor_list,  // [n_atoms * NEIGHBOR_LIST_SIZE]
    const int* __restrict__ n_neighbors,    // [n_atoms]
    // Excited state for charge modification
    const int* __restrict__ d_atom_to_aromatic,
    const int* __restrict__ d_aromatic_type,
    const int* __restrict__ d_is_excited,
    const float* __restrict__ d_electronic_population,
    const float* __restrict__ d_ground_state_charges,
    int n_atoms,
    float cutoff_sq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_atoms) return;

    float3 my_pos = positions[tid];
    float my_charge = charges[tid];
    LJParam my_lj = lj_params[tid];

    // Apply excited state charge scaling if applicable
    int arom_idx = d_atom_to_aromatic[tid];
    if (arom_idx >= 0 && d_is_excited[arom_idx]) {
        float pop = d_electronic_population[arom_idx];
        // CANONICAL: 0=TRP, 1=TYR, 2=PHE, 3=S-S
        float ratio_sqrt;
        switch (d_aromatic_type[arom_idx]) {
            case 0: ratio_sqrt = TRP_DIPOLE_RATIO_SQRT; break;  // TRP
            case 1: ratio_sqrt = TYR_DIPOLE_RATIO_SQRT; break;  // TYR
            case 2: ratio_sqrt = PHE_DIPOLE_RATIO_SQRT; break;  // PHE
            default: ratio_sqrt = 1.0f;
        }
        float scale = 1.0f + (ratio_sqrt - 1.0f) * pop;
        my_charge = d_ground_state_charges[tid] * scale;
    }

    float3 my_force = make_float3(0.0f, 0.0f, 0.0f);

    int my_n_neighbors = n_neighbors[tid];
    const int* my_neighbors = &neighbor_list[tid * NEIGHBOR_LIST_SIZE];

    // Only loop over actual neighbors (O(N) total work)
    for (int k = 0; k < my_n_neighbors; k++) {
        int j = my_neighbors[k];

        float3 other_pos = positions[j];
        float dx = my_pos.x - other_pos.x;
        float dy = my_pos.y - other_pos.y;
        float dz = my_pos.z - other_pos.z;
        float r2 = dx * dx + dy * dy + dz * dz;

        // Skip if outside cutoff (neighbor list has buffer)
        if (r2 >= cutoff_sq || r2 < 0.01f) continue;

        float r = sqrtf(r2);
        float inv_r = 1.0f / r;

        // Get other atom's charge with excited state scaling
        float other_charge = charges[j];
        int other_arom = d_atom_to_aromatic[j];
        if (other_arom >= 0 && d_is_excited[other_arom]) {
            float pop = d_electronic_population[other_arom];
            // CANONICAL: 0=TRP, 1=TYR, 2=PHE, 3=S-S
            float ratio_sqrt;
            switch (d_aromatic_type[other_arom]) {
                case 0: ratio_sqrt = TRP_DIPOLE_RATIO_SQRT; break;  // TRP
                case 1: ratio_sqrt = TYR_DIPOLE_RATIO_SQRT; break;  // TYR
                case 2: ratio_sqrt = PHE_DIPOLE_RATIO_SQRT; break;  // PHE
                default: ratio_sqrt = 1.0f;
            }
            float scale = 1.0f + (ratio_sqrt - 1.0f) * pop;
            other_charge = d_ground_state_charges[j] * scale;
        }

        // Lorentz-Berthelot combining rules
        float sigma_ij = 0.5f * (my_lj.sigma + lj_params[j].sigma);
        float eps_ij = sqrtf(my_lj.epsilon * lj_params[j].epsilon);

        // LJ 12-6 with soft core
        float r2_soft = r2 + 0.01f;  // Soft core delta
        float sigma2 = sigma_ij * sigma_ij;
        float sigma6 = sigma2 * sigma2 * sigma2;
        float inv_r2_soft = 1.0f / r2_soft;
        float inv_r6_soft = inv_r2_soft * inv_r2_soft * inv_r2_soft;
        float sigma6_r6 = sigma6 * inv_r6_soft;

        float lj_force = 24.0f * eps_ij * (2.0f * sigma6_r6 * sigma6_r6 - sigma6_r6) / r2_soft;

        // Coulomb with implicit solvent (ε = 4r)
        float coul_force = COULOMB_CONSTANT * my_charge * other_charge * 0.25f * inv_r * inv_r * inv_r;

        // Total force with capping
        float total_force = lj_force + coul_force;
        float max_force = 500.0f;
        if (fabsf(total_force) > max_force) {
            total_force = copysignf(max_force, total_force);
        }

        // Accumulate force on my atom
        my_force.x -= total_force * dx;
        my_force.y -= total_force * dy;
        my_force.z -= total_force * dz;

        // The production neighbor list is full and sorted per atom, so the
        // partner atom computes its own equal/opposite contribution in its own
        // thread. Cross-atom atomicAdd is intentionally avoided here.
    }

    forces[tid].x += my_force.x;
    forces[tid].y += my_force.y;
    forces[tid].z += my_force.z;
}

// ============================================================================
// EXTERN "C" INITIALIZATION KERNELS FOR CRYO-THERMAL DETECTION
// These initialize the state for the multi-modal sensitive detector
// ============================================================================

// Initialize multi-modal detector state for cryo probing
extern "C" __global__ void init_multimodal_detector(
    MultiModalVoxelState* state,
    int n_voxels,
    float baseline_temp
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_voxels) return;
    
    state[tid].thermal_potential = 0.0f;
    state[tid].thermal_baseline = baseline_temp;
    state[tid].thermal_spike_count = 0;
    
    state[tid].gradient_potential = 0.0f;
    state[tid].last_gradient_dir = make_float3(0.0f, 0.0f, 0.0f);
    state[tid].gradient_spike_count = 0;
    
    state[tid].melt_potential = 0.0f;
    state[tid].ice_fraction = 1.0f;  // Start frozen
    state[tid].melt_spike_count = 0;
    
    state[tid].combined_potential = 0.0f;
    state[tid].last_spike_time = 0;
    state[tid].in_refractory = false;
    
    state[tid].signal_to_noise = 1.0f;
    state[tid].confidence = 0.0f;
}

// Initialize thermal voxels for cryo-thermal detection
extern "C" __global__ void init_thermal_voxels(
    ThermalVoxel* voxels,
    int n_voxels,
    float initial_temp
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_voxels) return;
    
    voxels[tid].temperature = initial_temp;
    voxels[tid].baseline_temp = initial_temp;
    voxels[tid].delta_temp = 0.0f;
    voxels[tid].temp_gradient_mag = 0.0f;
    voxels[tid].temp_gradient_dir = make_float3(0.0f, 0.0f, 0.0f);
    
    voxels[tid].ice_fraction = (initial_temp < 273.15f) ? 1.0f : 0.0f;
    voxels[tid].melt_rate = 0.0f;
    
    // Ice has different thermal properties than water
    if (initial_temp < 273.15f) {
        voxels[tid].heat_capacity = 2.09f;       // J/(g·K) for ice
        voxels[tid].thermal_conductivity = 2.2f; // W/(m·K) for ice
    } else {
        voxels[tid].heat_capacity = 4.18f;       // J/(g·K) for water
        voxels[tid].thermal_conductivity = 0.6f; // W/(m·K) for water
    }
    
    
    voxels[tid].last_spike_time = 0;
    voxels[tid].lif_potential = 0.0f;
    voxels[tid].in_refractory = false;
}
// REMOVED: self-include (circular recursion)

extern "C" __global__ void nhs_voxel_step(
    // Positions (read-only, updated by atom kernel)
    const float3* __restrict__ positions,
    int n_atoms,
    // Voxel grid params
    int grid_dim,
    float grid_spacing,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    // Exclusion field arrays
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    // Warp matrix
    const WarpEntry* warp_matrix,
    // Atom metadata
    const int* __restrict__ atom_types,
    const float* __restrict__ charges,
    const int* __restrict__ residue_ids,
    // Aromatic data
    const float3* __restrict__ d_aromatic_centroids,
    const float3* __restrict__ d_ring_normals,
    const int* __restrict__ d_is_excited,
    const float* __restrict__ d_electronic_population,
    const float* __restrict__ d_vibrational_energy,
    const float* __restrict__ d_time_since_excitation,
    const int* __restrict__ d_aromatic_type,
    const int* __restrict__ d_atom_to_aromatic,
    int n_aromatics,
    // PRISM-TWIN v3.0: GPU-resident protocol state
    const ProtocolState* __restrict__ d_protocol,
    float* d_uv_signal_prev,
    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    // EFP arrays
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    // Aromatic neighbors (for expanded exclusion)
    const AromaticNeighbors* __restrict__ d_aromatic_neighbors,
    const float* __restrict__ d_franck_condon_progress,
    int* spike_grid_efp,             // independent EFP refractory grid
    // Signal preservation buffers (accumulated across all timesteps)
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    int* residue_step_causal               // [n_residues] KCC per-step causal counter
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = grid_dim * grid_dim * grid_dim;
    if (tid >= total_voxels) return;

    int v = tid;

    // Load dynamic state from ProtocolState into registers
    const float uv_wavelength_nm = d_protocol->uv_wavelength_nm;
    const int uv_burst_active = d_protocol->uv_burst_active;
    const float target_temp = d_protocol->current_temperature;
    const float dt = d_protocol->dt;
    const int timestep = (int)d_protocol->current_step;

    float3 grid_origin = make_float3(grid_origin_x, grid_origin_y, grid_origin_z);

    // Voxel coordinates (computed ONCE, shared across all phases)
    int vz = v / (grid_dim * grid_dim);
    int vy = (v / grid_dim) % grid_dim;
    int vx = v % grid_dim;
    float3 voxel_center = make_float3(
        grid_origin.x + (vx + 0.5f) * grid_spacing,
        grid_origin.y + (vy + 0.5f) * grid_spacing,
        grid_origin.z + (vz + 0.5f) * grid_spacing
    );

    // ====================================================================
    // FUSED PHASE 4: EXCLUSION FIELD + WATER DENSITY
    // ====================================================================
    water_density_prev[v] = water_density[v];

    WarpEntry entry = warp_matrix[v];
    float total_exclusion = 0.0f;
    // SPARSE VOXEL EARLY EXIT: skip voxels with no nearby atoms
    if (entry.n_atoms == 0) {
        // Still need to decay refractory counter and LIF potential
        if (spike_grid[v] > 0) spike_grid[v]--;
        if (spike_grid_efp[v] > 0) spike_grid_efp[v]--;
        lif_potential[v] *= 0.9f;  // passive decay
        efp_lif_potential[v] *= 0.8f;
        return;
    }
    float polar_field = 0.0f;

    for (int i = 0; i < entry.n_atoms; i++) {
        int a = entry.atom_indices[i];
        if (a < 0 || a >= n_atoms) continue;

        float contrib = compute_exclusion_contribution(
            positions[a], voxel_center,
            atom_types[a], charges[a]
        );

        if (n_aromatics > 0 && d_aromatic_centroids != nullptr) {
            float expanded_modifier = compute_expanded_exclusion_modifier(
                positions[a],
                d_aromatic_centroids,
                d_ring_normals,
                d_is_excited,
                d_electronic_population,
                n_aromatics
            );
            contrib *= expanded_modifier;
        } else {
            int aromatic_idx = d_atom_to_aromatic[a];
            if (aromatic_idx >= 0 && aromatic_idx < n_aromatics) {
                float excitation_modifier = get_exclusion_modifier(
                    aromatic_idx,
                    d_is_excited,
                    d_electronic_population
                );
                contrib *= excitation_modifier;
            }
        }

        total_exclusion += contrib * entry.atom_weights[i] * 4.0f;

        if (atom_types[a] == 1 || atom_types[a] == 2 || atom_types[a] == 3) {
            polar_field += contrib * 0.5f;
        }
    }

    total_exclusion = fminf(1.0f, total_exclusion);
    exclusion_field[v] = total_exclusion;
    water_density[v] = infer_water_density(total_exclusion, polar_field, target_temp);

    // ====================================================================
    // FUSED PHASE 5: NEUROMORPHIC LIF + UV-LIF COUPLING
    // ====================================================================

    // Refractory check — if counting down, decrement and skip LIF+EFP
    if (spike_grid_efp[v] > 0) spike_grid_efp[v]--;
    if (spike_grid[v] > 0) {
        spike_grid[v]--;
        // Still do EFP below (it has its own refractory check)
        goto efp_phase;
    }

    {
        float spike_intensity = 0.0f;
        float tau_mem = 0.1f;

        float uv_signal = 0.0f;
        int n_nearby_excited = 0;
        float min_distance_to_excited = 1000.0f;
        int closest_excited_idx = -1;

        if (n_aromatics > 0) {
            const float UV_DETECTION_RADIUS = 4.0f;
            const float UV_DIRECT_STRENGTH = 0.8f;

            float total_vib_energy = 0.0f;

            for (int a = 0; a < n_aromatics; a++) {
                if (!d_is_excited[a]) continue;
                if (d_aromatic_centroids == nullptr) continue;

                float3 arom_pos = d_aromatic_centroids[a];
                float dx = voxel_center.x - arom_pos.x;
                float dy = voxel_center.y - arom_pos.y;
                float dz = voxel_center.z - arom_pos.z;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);

                if (dist < UV_DETECTION_RADIUS) {
                    n_nearby_excited++;
                    total_vib_energy += d_vibrational_energy[a];
                    if (dist < min_distance_to_excited) {
                        min_distance_to_excited = dist;
                        closest_excited_idx = a;
                    }
                }
            }

            if (n_nearby_excited > 0) {
                float energy_factor = fminf(total_vib_energy / (n_nearby_excited * 3.0f), 1.0f);
                float coop_boost = 1.0f + 0.3f * (n_nearby_excited - 1);
                uv_signal = UV_DIRECT_STRENGTH * energy_factor * coop_boost;
                if (uv_burst_active) {
                    uv_signal *= 2.0f;
                }
            }

            if (d_aromatic_centroids != nullptr && d_uv_signal_prev != nullptr) {
                float prev_signal = d_uv_signal_prev[v];
                float advanced_signal = compute_uv_lif_signal(
                    voxel_center,
                    d_aromatic_centroids,
                    d_ring_normals,
                    d_is_excited,
                    d_electronic_population,
                    d_vibrational_energy,
                    d_time_since_excitation,
                    n_aromatics,
                    dt,
                    prev_signal
                );
                uv_signal += advanced_signal;
                d_uv_signal_prev[v] = uv_signal;
            }

            if (uv_signal > 0.0f) {
                lif_potential[v] += uv_signal;
            }

            // Record UV event timestamp for causal tracking (inline nhs_voxel_step)
            if (uv_signal > 0.1f) {
                last_uv_step[v] = timestep;
            }

            // Direct UV spike trigger
            const float DIRECT_UV_SPIKE_THRESHOLD = 0.3f;
            const float MAX_SPIKE_DISTANCE = 4.0f;

            const int voxel_n_atoms = warp_matrix[v].n_atoms;
            bool voxel_has_aromatic_atom = false;
            for (int wi = 0; wi < voxel_n_atoms && !voxel_has_aromatic_atom; wi++) {
                int atom_idx = warp_matrix[v].atom_indices[wi];
                if (atom_idx >= 0 && atom_idx < n_atoms) {
                    if (d_atom_to_aromatic[atom_idx] >= 0) {
                        voxel_has_aromatic_atom = true;
                    }
                }
            }

            if (n_nearby_excited > 0 &&
                uv_signal > DIRECT_UV_SPIKE_THRESHOLD &&
                voxel_n_atoms > 0 &&
                min_distance_to_excited < MAX_SPIKE_DISTANCE) {
                spike_grid[v] = REFRACTORY_STEPS;
                spike_intensity = uv_signal;

                int _arom_type = (closest_excited_idx >= 0) ? d_aromatic_type[closest_excited_idx] : -1;
                int _arom_res = -1;
                if (closest_excited_idx >= 0) {
                    for (int wi = 0; wi < warp_matrix[v].n_atoms && _arom_res < 0; wi++) {
                        int ai = warp_matrix[v].atom_indices[wi];
                        if (ai >= 0 && ai < n_atoms && d_atom_to_aromatic[ai] == closest_excited_idx) {
                            _arom_res = residue_ids[ai];
                        }
                    }
                }
                float _vib_e = (closest_excited_idx >= 0) ? d_vibrational_energy[closest_excited_idx] : 0.0f;

                // Signal preservation (late UV spike)
                update_signal_preservation(v, timestep, 1,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[v], residue_ids, n_atoms, residue_step_causal);

                int spike_idx = atomicAdd(spike_count, 1);
                if (spike_idx < max_spikes) {
                    capture_spike_event(
                        spike_events[spike_idx], timestep, v, voxel_center,
                        spike_intensity, warp_matrix[v], residue_ids,
                        1, uv_wavelength_nm, _arom_type, _arom_res,
                        water_density[v], _vib_e, n_nearby_excited,
                        fabsf(water_density[v] - water_density_prev[v]),
                        d_protocol->current_phase_bits
                    );
                }
                lif_potential[v] = LIF_RESET;
            }
        }

        // Standard LIF update (non-UV)
        if (spike_grid[v] == 0 && !uv_burst_active) {
            bool spike = lif_neuron_update(
                lif_potential[v], water_density[v], water_density_prev[v],
                tau_mem, dt, LIF_THRESHOLD, spike_intensity
            );

            if (spike) {
                spike_grid[v] = REFRACTORY_STEPS;

                // Signal preservation (late LIF spike)
                int lif_source2 = (n_nearby_excited > 0) ? 1 : 2;
                update_signal_preservation(v, timestep, lif_source2,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[v], residue_ids, n_atoms, residue_step_causal);

                int spike_idx = atomicAdd(spike_count, 1);
                if (spike_idx < max_spikes) {
                    int lif_atype = -1, lif_ares = -1;
                    float lif_wl = 0.0f, lif_vibe = 0.0f;
                    if (n_nearby_excited > 0 && closest_excited_idx >= 0) {
                        lif_atype = d_aromatic_type[closest_excited_idx];
                        lif_wl = uv_wavelength_nm;
                        lif_vibe = d_vibrational_energy[closest_excited_idx];
                        for (int wi = 0; wi < warp_matrix[v].n_atoms && lif_ares < 0; wi++) {
                            int ai = warp_matrix[v].atom_indices[wi];
                            if (ai >= 0 && ai < n_atoms && d_atom_to_aromatic[ai] == closest_excited_idx) {
                                lif_ares = residue_ids[ai];
                            }
                        }
                    }
                    capture_spike_event(
                        spike_events[spike_idx], timestep, v, voxel_center,
                        spike_intensity, warp_matrix[v], residue_ids,
                        lif_source2, lif_wl,
                        lif_atype, lif_ares, water_density[v],
                        lif_vibe, n_nearby_excited,
                        fabsf(water_density[v] - water_density_prev[v]),
                        d_protocol->current_phase_bits
                    );
                }
            }
        }
    }

    // ====================================================================
    // FUSED EFP: ELECTROSTATIC FLUX PROBE
    // ====================================================================
efp_phase:

    if (warp_matrix[v].n_atoms > 0) {
        float phi = 0.0f;
        int n_charged_nearby = 0;

        for (int wi = 0; wi < warp_matrix[v].n_atoms; wi++) {
            int ai = warp_matrix[v].atom_indices[wi];
            if (ai < 0 || ai >= n_atoms) continue;
            float q = charges[ai];
            if (fabsf(q) < 0.15f) continue;

            float3 ap = positions[ai];
            float dx = voxel_center.x - ap.x;
            float dy = voxel_center.y - ap.y;
            float dz = voxel_center.z - ap.z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);

            if (dist > 0.5f && dist < 8.0f) {
                float eps_r = fmaxf(4.0f * dist, 4.0f);
                phi += q / (eps_r * dist);
                n_charged_nearby++;
            }
        }

        float wd_change = fabsf(water_density[v] - water_density_prev[v]);
        float polar_water_signal = fabsf(phi) * wd_change * 40.0f;

        float phi_prev = efp_potential_prev[v];
        efp_potential[v] = phi;

        float flux = fabsf(phi - phi_prev);
        float polar_signal = flux * 150.0f + polar_water_signal;

        if (n_charged_nearby >= 1 && spike_grid_efp[v] == 0) {
            const float EFP_TAU = 0.5f;
            const float EFP_THRESHOLD = 0.15f;
            float efp_decay = expf(-dt / EFP_TAU);
            efp_lif_potential[v] = efp_decay * efp_lif_potential[v] + polar_signal;

            if (efp_lif_potential[v] > EFP_THRESHOLD) {
                spike_grid_efp[v] = REFRACTORY_STEPS;
                float polar_intensity = efp_lif_potential[v];
                efp_lif_potential[v] = LIF_RESET;

                // Signal preservation (late EFP spike)
                update_signal_preservation(v, timestep, 3,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[v], residue_ids, n_atoms, residue_step_causal);

                int si = atomicAdd(spike_count, 1);
                if (si < max_spikes) {
                    int polar_type = (phi > 0.0f) ? 5 : 6;
                    capture_spike_event(
                        spike_events[si], timestep, v, voxel_center,
                        polar_intensity, warp_matrix[v], residue_ids,
                        3, 0.0f, polar_type, -1,
                        water_density[v], flux, n_charged_nearby,
                        wd_change,  // already computed above
                        d_protocol->current_phase_bits
                    );
                }
            }
        }
        efp_potential_prev[v] = phi;
    }
}

// ============================================================================
// MULTI-NEURON RAF (Resonant Adaptive Filter) WITH SHARED MEMORY STENCIL
// ============================================================================
//
// K=8 RAF oscillator neurons per voxel, 8 threads per voxel, 3D tile launch.
// Each neuron has a different resonant frequency (omega_k = 2*pi/tau_k).
// Cross-voxel coupling via shared-memory halo stencil (26 Moore neighbors).
//
// Neuron timescales: tau_k = 0.5 * 2^k  (k=0..7)
//   k=0: 0.5 ps   (ultrafast electronic)
//   k=1: 1.0 ps   (bond vibration)
//   k=2: 2.0 ps   (hydrogen bond)
//   k=3: 4.0 ps   (sidechain rotation)
//   k=4: 8.0 ps   (loop motion)
//   k=5: 16.0 ps  (domain hinge)
//   k=6: 32.0 ps  (slow collective)
//   k=7: 64.0 ps  (cryptic site opening)
//
// State arrays (no new buffers — repurposed):
//   neuron_potential[nidx] = x (real part of oscillator)
//   neuron_mean[nidx]      = y (imaginary part of oscillator)
//   neuron_threshold[nidx] = adaptive threshold
//   neuron_refractory[nidx]= refractory counter
// ============================================================================

#define K_NEURONS 8
#define THREADS_PER_VOXEL 8
#define VOXELS_PER_WARP (WARP_SIZE / THREADS_PER_VOXEL)  // 4
#define MULTI_LIF_THRESHOLD 0.3f
#define COUPLING_STRENGTH 0.01f
#define COUPLING_DECAY 0.9f

// 3D tile dimensions for shared memory stencil
// Reduced from 4x4x2 (256 threads) to 4x2x2 (128 threads) for better
// SM occupancy on Blackwell (~25% vs ~12.5% at 256 threads/block).
#define TILE_X 4
#define TILE_Y 2
#define TILE_Z 2
#define VOXELS_PER_TILE (TILE_X * TILE_Y * TILE_Z)  // 16
#define HALO_X (TILE_X + 2)   // 6
#define HALO_Y (TILE_Y + 2)   // 4
#define HALO_Z (TILE_Z + 2)   // 4
#define HALO_SIZE (HALO_X * HALO_Y * HALO_Z)  // 96

// RAF parameters
#define CASCADE_RATE 0.01f
#define JITTER_AMPLITUDE 0.02f
#define RAF_R_MAX 2.0f
#define RAF_PI 3.14159265f

// Compute neuron timescale from lane index within voxel
__device__ __forceinline__ float neuron_tau(int k) {
    // tau = 0.5 * 2^k = 0.5, 1, 2, 4, 8, 16, 32, 64
    return 0.5f * (float)(1 << k);
}

// Compute 1D voxel index from 3D coordinates (clamped)
__device__ __forceinline__ int voxel_idx_3d(int x, int y, int z, int gdim) {
    if (x < 0 || x >= gdim || y < 0 || y >= gdim || z < 0 || z >= gdim) return -1;
    return z * gdim * gdim + y * gdim + x;
}

extern "C" __global__ __launch_bounds__(128, 8) void nhs_voxel_step_multi_lif(
    // === Same as nhs_voxel_step ===
    const float3* __restrict__ positions,
    int n_atoms,
    int grid_dim,
    float grid_spacing,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    int* spike_grid,
    const WarpEntry* warp_matrix,
    const int* __restrict__ atom_types,
    const float* __restrict__ charges,
    const int* __restrict__ residue_ids,
    const float3* __restrict__ d_aromatic_centroids,
    const float3* __restrict__ d_ring_normals,
    const int* __restrict__ d_is_excited,
    const float* __restrict__ d_electronic_population,
    const float* __restrict__ d_vibrational_energy,
    const float* __restrict__ d_time_since_excitation,
    const int* __restrict__ d_aromatic_type,
    const int* __restrict__ d_atom_to_aromatic,
    int n_aromatics,
    // PRISM-TWIN v3.0: GPU-resident protocol state
    const ProtocolState* __restrict__ d_protocol,
    float* d_uv_signal_prev,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    const AromaticNeighbors* __restrict__ d_aromatic_neighbors,
    const float* __restrict__ d_franck_condon_progress,
    // === Multi-neuron buffers ===
    float* neuron_potential,     // [total_voxels * K_NEURONS] = x (real)
    float* neuron_threshold,     // [total_voxels * K_NEURONS]
    float* neuron_mean,          // [total_voxels * K_NEURONS] = y (imaginary)
    int*   neuron_refractory,    // [total_voxels * K_NEURONS]
    float* coupling_a,           // [total_voxels] double-buffer A
    float* coupling_b,           // [total_voxels] double-buffer B
    // Director toggles coupling_phase at each observation boundary.
    // phase=0: read A, write B.  phase=1: read B, write A.
    // === Sparse tile index ===
    const int* active_tiles,     // [n_active_tiles * 3] packed (bx, by, bz) triplets
    int n_active_tiles,          // number of active tiles (0 = fallback to full grid)
    // === Independent EFP refractory grid ===
    int* spike_grid_efp,         // [grid_dim³] — prevents UV/LIF from blocking EFP
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal,              // [n_residues] zeroed each step by kcc_residue_update
    // === LADD: protein density reference (appended, backward-safe) ===
    float* ladd_density_ref,               // [total_voxels] cold-phase mean protein density
    float* ladd_density_m2,                // [total_voxels] Welford M2 for variance
    int* ladd_ref_count,                   // [total_voxels] reference sample count
    int ladd_enabled,                      // 0 or 1
    int ladd_cold_hold_steps               // steps for reference accumulation
) {
    // Uniform across the entire grid and before any shared-memory barrier.
    // The Director/physics/guard still execute on every resident iteration.
    if (!prism_protocol_observation_due(d_protocol)) return;

    // Load dynamic state from ProtocolState into registers
    const float uv_wavelength_nm = d_protocol->uv_wavelength_nm;
    const int uv_burst_active = d_protocol->uv_burst_active;
    const float target_temp = d_protocol->current_temperature;
    const float dt = d_protocol->dt;
    const int timestep = (int)d_protocol->current_step;

    // Coupling buffer self-selection: Director already toggled coupling_phase.
    // The observation kernel reads the current buffer; the deterministic
    // coupling builder writes the alternate buffer in its separate phase.
    const float* coupling_read = (d_protocol->coupling_phase == 0) ? coupling_a : coupling_b;

    // === Shared memory layout (SOTA v2) ===
    // [0..HALO_SIZE-1]:  coupling stencil halo tile (96 floats)
    // [HALO_SIZE..+7]:   cos(omega*dt) LUT per neuron k (8 floats)
    // [+8..+15]:         sin(omega*dt) LUT (8 floats)
    // [+16..+23]:        exp(-dt/tau) LUT (8 floats)
    // [+24]:             flag: any aromatic excited (1 int)
    // [+25]:             n_excited_cached (1 int)
    // [+26..+26+MAX_AROM*10-1]: CachedAromaticData array
    extern __shared__ float s_coupling_tile[];
    float* s_cos_lut   = &s_coupling_tile[HALO_SIZE];       // [8]
    float* s_sin_lut   = &s_coupling_tile[HALO_SIZE + 8];   // [8]
    float* s_decay_lut = &s_coupling_tile[HALO_SIZE + 16];  // [8]
    int*   s_any_excited = (int*)&s_coupling_tile[HALO_SIZE + 24]; // [1]
    int*   s_n_excited   = (int*)&s_coupling_tile[HALO_SIZE + 25]; // [1]
    // Aromatic cache: 10 floats per aromatic (float3 pos=3, float3 norm=3, epop, vibe, tse, is_excited)
    #define SMEM_AROM_OFFSET (HALO_SIZE + 26)
    #define MAX_AROM_CACHED 64
    float* s_arom_cache = &s_coupling_tile[SMEM_AROM_OFFSET]; // [MAX_AROM_CACHED * 10]

    // Thread 0..7: build trig/decay LUT
    if (threadIdx.x < K_NEURONS) {
        float my_tau = neuron_tau(threadIdx.x);
        float omega = 2.0f * RAF_PI / my_tau;
        s_cos_lut[threadIdx.x]   = cosf(omega * dt);
        s_sin_lut[threadIdx.x]   = sinf(omega * dt);
        s_decay_lut[threadIdx.x] = expf(-dt / my_tau);
    }

    // Cooperative aromatic cache load: all 128 threads load aromatic data
    // into shared memory. Eliminates repeated global reads in exclusion + UV loops.
    {
        int n_arom = min(n_aromatics, MAX_AROM_CACHED);
        int n_excited_count = 0;
        // Each thread loads a subset of aromatics (10 floats per aromatic)
        for (int a = threadIdx.x; a < n_arom; a += blockDim.x) {
            float* dst = &s_arom_cache[a * 10];
            float3 pos = d_aromatic_centroids[a];
            float3 nrm = d_ring_normals[a];
            dst[0] = pos.x;  dst[1] = pos.y;  dst[2] = pos.z;
            dst[3] = nrm.x;  dst[4] = nrm.y;  dst[5] = nrm.z;
            dst[6] = d_electronic_population[a];
            dst[7] = d_vibrational_energy[a];
            dst[8] = d_time_since_excitation[a];
            dst[9] = __int_as_float(d_is_excited[a]);
        }
        __syncthreads();
        // Thread 0: count excited aromatics (from shared memory, not global)
        if (threadIdx.x == 0) {
            int any = 0;
            for (int a = 0; a < n_arom; a++) {
                if (__float_as_int(s_arom_cache[a * 10 + 9])) { any = 1; n_excited_count++; }
            }
            *s_any_excited = any;
            *s_n_excited = n_excited_count;
        }
    }
    __syncthreads();

    // === Sparse tile mapping ===
    // Each blockIdx.x maps to an active tile via the sparse index.
    // If n_active_tiles == 0, fall back to linear decomposition (full grid).
    int tile_bx, tile_by, tile_bz;
    if (n_active_tiles > 0 && active_tiles != nullptr) {
        int tile_idx = blockIdx.x;
        tile_bx = active_tiles[tile_idx * 3 + 0];
        tile_by = active_tiles[tile_idx * 3 + 1];
        tile_bz = active_tiles[tile_idx * 3 + 2];
    } else {
        // Fallback: reconstruct 3D from linear blockIdx
        int tiles_x = (grid_dim + TILE_X - 1) / TILE_X;
        int tiles_y = (grid_dim + TILE_Y - 1) / TILE_Y;
        tile_bx = blockIdx.x % tiles_x;
        tile_by = (blockIdx.x / tiles_x) % tiles_y;
        tile_bz = blockIdx.x / (tiles_x * tiles_y);
    }

    int local_voxel = threadIdx.x / THREADS_PER_VOXEL;  // 0..15
    int k = threadIdx.x % THREADS_PER_VOXEL;             // 0..7

    // Local 3D within tile
    int lz = local_voxel / (TILE_X * TILE_Y);
    int ly = (local_voxel / TILE_X) % TILE_Y;
    int lx = local_voxel % TILE_X;

    // Global 3D voxel coordinates
    int vx = tile_bx * TILE_X + lx;
    int vy = tile_by * TILE_Y + ly;
    int vz = tile_bz * TILE_Z + lz;

    bool is_valid = (vx < grid_dim && vy < grid_dim && vz < grid_dim);
    int v = is_valid ? (vz * grid_dim * grid_dim + vy * grid_dim + vx) : 0;

    float3 grid_origin = make_float3(grid_origin_x, grid_origin_y, grid_origin_z);
    float3 voxel_center = make_float3(
        grid_origin.x + (vx + 0.5f) * grid_spacing,
        grid_origin.y + (vy + 0.5f) * grid_spacing,
        grid_origin.z + (vz + 0.5f) * grid_spacing
    );

    // Warp-level lane info
    unsigned int warp_mask = 0xFFFFFFFF;
    int lane_in_warp = threadIdx.x % WARP_SIZE;
    int voxel_in_warp = lane_in_warp / THREADS_PER_VOXEL;  // 0..3
    int src_lane = voxel_in_warp * THREADS_PER_VOXEL;
    unsigned int voxel_mask = 0xFFu << (voxel_in_warp * THREADS_PER_VOXEL);

    // Initialize shared memory halo to zero
    for (int i = threadIdx.x; i < HALO_SIZE; i += blockDim.x) {
        s_coupling_tile[i] = 0.0f;
    }
    __syncthreads();

    // ====================================================================
    // PHASE 1: EXCLUSION FIELD + WATER DENSITY (neuron 0 only)
    // ====================================================================
    bool has_atoms = false;
    WarpEntry entry;
    int nidx = is_valid ? (v * K_NEURONS + k) : 0;

    if (is_valid) {
        water_density_prev[v] = water_density[v];  // all 8 threads idempotent
        entry = warp_matrix[v];
        has_atoms = (entry.n_atoms > 0);

        // Sparse voxel: passive decay only
        if (!has_atoms) {
            if (k == 0 && spike_grid[v] > 0) spike_grid[v]--;
            neuron_potential[nidx] *= 0.9f;
            neuron_mean[nidx] *= 0.9f;  // y component also decays
            if (k == 0) efp_lif_potential[v] *= 0.8f;
        }
    }

    // ────────────────────────────────────────────────────────────────
    // PHASE 1: WARP-COOPERATIVE EXCLUSION (all 8 neurons contribute)
    // Instead of k==0 serializing over atoms, all 8 threads process
    // atoms in parallel with warp shuffle reduction. 8x parallelism
    // on the most expensive per-voxel loop.
    // ────────────────────────────────────────────────────────────────
    float total_exclusion = 0.0f;
    float polar_field = 0.0f;
    float my_protein_density = 0.0f;  // LADD: heavy-atom density (wider scope for reference block)

    if (is_valid && has_atoms) {
        // All 8 neurons cooperatively process atoms: each neuron handles
        // atoms[k], atoms[k+8], atoms[k+16], ... (strided access)
        float my_exclusion = 0.0f;
        float my_polar = 0.0f;
        int n_arom = min(n_aromatics, MAX_AROM_CACHED);

        for (int i = k; i < entry.n_atoms; i += K_NEURONS) {
            int a = entry.atom_indices[i];
            if (a < 0 || a >= n_atoms) continue;

            float contrib = compute_exclusion_contribution(
                positions[a], voxel_center,
                atom_types[a], charges[a]
            );

            // LADD: accumulate protein heavy-atom density (Gaussian, sigma from LJ)
            {
                float3 apos = positions[a];
                float dx = apos.x - voxel_center.x;
                float dy = apos.y - voxel_center.y;
                float dz = apos.z - voxel_center.z;
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < 25.0f) {  // 5Å cutoff
                    my_protein_density += expf(-r2 * 0.5f);  // unit Gaussian, σ≈1Å
                }
            }

            // Aromatic exclusion modifier: read from shared memory cache
            if (n_arom > 0) {
                float best_mod = 1.0f;
                float3 atom_pos = positions[a];
                for (int ar = 0; ar < n_arom; ar++) {
                    float* ad = &s_arom_cache[ar * 10];
                    int excited = __float_as_int(ad[9]);
                    if (!excited) continue;
                    float dx = atom_pos.x - ad[0];
                    float dy = atom_pos.y - ad[1];
                    float dz = atom_pos.z - ad[2];
                    float dist_sq = dx*dx + dy*dy + dz*dz;
                    if (dist_sq < 25.0f) { // 5Å cutoff
                        float epop = ad[6];
                        float mod = 1.0f + epop * 0.5f;
                        if (mod > best_mod) best_mod = mod;
                    }
                }
                contrib *= best_mod;
            } else {
                int aromatic_idx = d_atom_to_aromatic[a];
                if (aromatic_idx >= 0 && aromatic_idx < n_aromatics) {
                    float excitation_modifier = get_exclusion_modifier(
                        aromatic_idx,
                        d_is_excited,
                        d_electronic_population
                    );
                    contrib *= excitation_modifier;
                }
            }

            my_exclusion += contrib * entry.atom_weights[i] * 4.0f;

            if (atom_types[a] == 1 || atom_types[a] == 2 || atom_types[a] == 3) {
                my_polar += contrib * 0.5f;
            }
        }

        // Warp shuffle reduction across K neurons within voxel
        for (int off = K_NEURONS / 2; off > 0; off >>= 1) {
            my_exclusion += __shfl_xor_sync(voxel_mask, my_exclusion, off);
            my_polar     += __shfl_xor_sync(voxel_mask, my_polar, off);
            my_protein_density += __shfl_xor_sync(voxel_mask, my_protein_density, off);
        }
        total_exclusion = fminf(1.0f, my_exclusion);
        polar_field = my_polar;

        // Neuron 0 writes results
        if (k == 0) {
            exclusion_field[v] = total_exclusion;
            water_density[v] = infer_water_density(total_exclusion, polar_field, target_temp);
        }
    }

    // All neurons already have total_exclusion and my_protein_density from shuffle

    // ====================================================================
    // LADD: Reference accumulation (cold_hold) + depletion computation
    // ====================================================================
    float ladd_input = 0.0f;  // protein density depletion signal for LADD neurons
    if (is_valid && has_atoms && ladd_enabled && k == 0) {
        float pd = my_protein_density;  // after warp reduce, all neurons have the sum
        if (timestep < ladd_cold_hold_steps) {
            // Cold-hold: accumulate Welford reference
            int count = ladd_ref_count[v] + 1;
            float delta = pd - ladd_density_ref[v];
            float new_mean = ladd_density_ref[v] + delta / (float)count;
            float delta2 = pd - new_mean;
            ladd_density_m2[v] += delta * delta2;
            ladd_density_ref[v] = new_mean;
            ladd_ref_count[v] = count;
        } else {
            // Post cold-hold: compute depletion signal
            float ref = ladd_density_ref[v];
            if (ref > 0.5f) {
                float depletion = (ref - pd) / ref;
                ladd_input = fmaxf(depletion, 0.0f);  // only positive (void forming)
            }
        }
    }
    // Broadcast ladd_input from neuron 0 to all neurons in the voxel
    ladd_input = __shfl_sync(warp_mask, ladd_input, src_lane);

    // ====================================================================
    // UV SIGNAL COMPUTATION (neuron 0 only, broadcast to all)
    // Must happen BEFORE refractory check so all threads reach __shfl_sync
    // ====================================================================
    float uv_signal = 0.0f;
    int n_nearby_excited = 0;
    float min_distance_to_excited = 1000.0f;
    int closest_excited_idx = -1;

    // ────────────────────────────────────────────────────────────────
    // UV SIGNAL: Shared-memory cached aromatics (zero global reads)
    // Single fused pass over cached aromatics replaces two separate
    // global-memory loops (basic UV + advanced compute_uv_lif_signal).
    // ────────────────────────────────────────────────────────────────
    if (is_valid && has_atoms && k == 0 && n_aromatics > 0 && *s_any_excited) {
        const float UV_DETECTION_RADIUS = 4.0f;
        const float UV_DIRECT_STRENGTH = 0.8f;
        const float MAX_DETECTION_RADIUS = 20.0f;
        float total_vib_energy = 0.0f;
        float thermal_wavefront_sum = 0.0f;
        float halo_sum = 0.0f;
        int n_arom = min(n_aromatics, MAX_AROM_CACHED);

        // Single fused loop: basic UV proximity + advanced thermal/halo
        #pragma unroll 4
        for (int a = 0; a < n_arom; a++) {
            float* ad = &s_arom_cache[a * 10];
            int excited = __float_as_int(ad[9]);
            if (!excited) continue;

            float ddx = voxel_center.x - ad[0];
            float ddy = voxel_center.y - ad[1];
            float ddz = voxel_center.z - ad[2];
            float dist_sq = ddx*ddx + ddy*ddy + ddz*ddz;

            if (dist_sq > MAX_DETECTION_RADIUS * MAX_DETECTION_RADIUS) continue;
            float dist = sqrtf(dist_sq);

            // Basic UV proximity signal (within 4Å)
            if (dist < UV_DETECTION_RADIUS) {
                n_nearby_excited++;
                total_vib_energy += ad[7]; // vibrational_energy from cache
                if (dist < min_distance_to_excited) {
                    min_distance_to_excited = dist;
                    closest_excited_idx = a;
                }
            }

            // Advanced thermal wavefront (within 20Å)
            float3 arom_pos = make_float3(ad[0], ad[1], ad[2]);
            float3 ring_norm = make_float3(ad[3], ad[4], ad[5]);
            float tse = ad[8]; // time_since_excitation
            float vibe = ad[7]; // vibrational_energy
            float epop = ad[6]; // electronic_population

            thermal_wavefront_sum += compute_thermal_wavefront(
                voxel_center, arom_pos, ring_norm, tse, vibe
            );
            halo_sum += compute_dewetting_halo(dist, epop);
        }

        // Basic UV signal
        if (n_nearby_excited > 0) {
            float energy_factor = fminf(total_vib_energy / (n_nearby_excited * 3.0f), 1.0f);
            float coop_boost = 1.0f + 0.3f * (n_nearby_excited - 1);
            uv_signal = UV_DIRECT_STRENGTH * energy_factor * coop_boost;
            if (uv_burst_active) uv_signal *= 2.0f;
        }

        // Advanced signal from fused thermal + halo (replaces compute_uv_lif_signal)
        if (d_uv_signal_prev != nullptr && (thermal_wavefront_sum > 0.0f || halo_sum > 0.0f)) {
            float prev_signal = d_uv_signal_prev[v];
            int n_contrib = n_nearby_excited > 0 ? n_nearby_excited : 1;
            float coop = compute_cooperative_boost(n_contrib);
            float advanced_signal = (thermal_wavefront_sum + halo_sum) * coop;
            // Temporal smoothing
            float smoothed = 0.7f * advanced_signal + 0.3f * prev_signal;
            uv_signal += smoothed;
            d_uv_signal_prev[v] = uv_signal;
        }
    }

    // Record UV event timestamp for causal tracking (neuron 0 only, before broadcast)
    if (is_valid && has_atoms && k == 0 && uv_signal > 0.1f) {
        last_uv_step[v] = timestep;
    }

    // Broadcast UV results from neuron 0 to all 8 neurons
    uv_signal = __shfl_sync(warp_mask, uv_signal, src_lane);
    n_nearby_excited = __shfl_sync(warp_mask, n_nearby_excited, src_lane);
    min_distance_to_excited = __shfl_sync(warp_mask, min_distance_to_excited, src_lane);
    closest_excited_idx = __shfl_sync(warp_mask, closest_excited_idx, src_lane);

    // ====================================================================
    // PHASE 2: RAF OSCILLATOR UPDATE
    // ====================================================================

    // Spike grid decrement (neuron 0 only)
    if (is_valid && has_atoms && k == 0 && spike_grid[v] > 0) {
        spike_grid[v]--;
    }
    if (is_valid && has_atoms && k == 0 && spike_grid_efp[v] > 0) {
        spike_grid_efp[v]--;
    }

    // Read coupling from previous step
    float coupling_input = (is_valid && has_atoms) ? coupling_read[v] : 0.0f;

    // Refractory check — per-neuron
    bool in_refractory = false;
    if (is_valid && has_atoms) {
        int refrac = neuron_refractory[nidx];
        if (refrac > 0) {
            neuron_refractory[nidx] = refrac - 1;
            in_refractory = true;
        }
    }

    // RAF oscillator state (initialized for all threads; only updated for active)
    float x_new = 0.0f, y_new = 0.0f;
    float threshold = MULTI_LIF_THRESHOLD;
    bool my_spike = false;

    if (is_valid && has_atoms && !in_refractory) {
        // Read oscillator state from global memory
        float x = neuron_potential[nidx];
        float y = neuron_mean[nidx];  // repurposed as imaginary part
        threshold = neuron_threshold[nidx];
        float my_tau = neuron_tau(k);

        // Compute water density signal (bidirectional + exclusion deviation)
        float wd_curr = water_density[v];
        float wd_prev = water_density_prev[v];
        float density_change = wd_curr - wd_prev;

        const float NOISE_FLOOR = 0.002f;
        float abs_change = fabsf(density_change);
        float bidirectional_signal = (abs_change > NOISE_FLOOR) ?
            (abs_change - NOISE_FLOOR) * 3.0f : 0.0f;

        float density_deviation = fabsf(wd_curr - WATER_DENSITY_BULK);
        const float EXCLUSION_NOISE_FLOOR = 0.005f;
        float exclusion_signal = (density_deviation > EXCLUSION_NOISE_FLOOR) ?
            (density_deviation - EXCLUSION_NOISE_FLOOR) * 2.0f : 0.0f;

        float combined_signal = bidirectional_signal + exclusion_signal + uv_signal;

        // RAF oscillator: damped harmonic oscillator with signal injection
        //   x_new = decay * (x*cos(wt) + y*sin(wt)) + signal
        //   y_new = decay * (-x*sin(wt) + y*cos(wt))
        // Read pre-computed trig from shared memory LUT (avoids 3 transcendentals per thread)
        float cos_wt = s_cos_lut[k];
        float sin_wt = s_sin_lut[k];
        float decay  = s_decay_lut[k];

        x_new = decay * (x * cos_wt + y * sin_wt) + combined_signal;
        y_new = decay * (-x * sin_wt + y * cos_wt);

        // Stochastic jitter (hash-based PRNG)
        unsigned int hash = (unsigned int)(v) * 2654435761u
                          + (unsigned int)(k) * 2246822519u
                          + (unsigned int)(timestep) * 3266489917u;
        hash ^= hash >> 16;
        hash *= 0x45d9f3bu;
        hash ^= hash >> 16;
        float jitter = JITTER_AMPLITUDE * ((float)(hash & 0xFFFF) / 65536.0f - 0.5f);
        x_new += jitter;
    }

    // Inter-neuron energy cascade: fast neurons (low k) feed slow neurons (high k)
    // All threads participate in warp shuffle (refractory/invalid contribute amp=0)
    float my_amp = sqrtf(x_new * x_new + y_new * y_new);
    float prev_amp = __shfl_up_sync(warp_mask, my_amp, 1);
    if (is_valid && has_atoms && !in_refractory && k > 0) {
        x_new += CASCADE_RATE * prev_amp;
    }

    // Continue RAF update for active neurons: saturated clamp + spike check
    if (is_valid && has_atoms && !in_refractory) {
        // Saturated resonance clamp
        float r2 = x_new * x_new + y_new * y_new;
        if (r2 > RAF_R_MAX * RAF_R_MAX) {
            float scale = RAF_R_MAX * rsqrtf(fmaxf(r2, 1e-12f));
            x_new *= scale;
            y_new *= scale;
            r2 = RAF_R_MAX * RAF_R_MAX;
        }

        // Spike check — coupling modulates threshold (NOT signal)
        float coupling_norm = fminf(coupling_input, 5.0f);
        float effective_threshold = threshold - COUPLING_STRENGTH * coupling_norm;
        if (effective_threshold < 0.2f) effective_threshold = 0.2f;

        float amplitude = sqrtf(r2);
        // RAF spikes require UV contribution — prevents crowding out EFP
        my_spike = (amplitude >= effective_threshold);

        if (my_spike) {
            // Always reset oscillator on threshold crossing
            x_new = 0.01f;
            y_new = 0.0f;
            // Only enter refractory + raise threshold when spike can be emitted
            if (spike_grid[v] == 0) {
                neuron_refractory[nidx] = REFRACTORY_STEPS;
                threshold = fminf(threshold + 0.02f, 2.0f * MULTI_LIF_THRESHOLD);
            }
        } else {
            threshold = fmaxf(threshold - 0.001f * dt, MULTI_LIF_THRESHOLD * 0.5f);
        }

        // Write back oscillator state
        neuron_potential[nidx] = x_new;
        neuron_mean[nidx] = y_new;
        neuron_threshold[nidx] = threshold;
    }

    // ====================================================================
    // SPIKE AGGREGATION + SHARED MEMORY STENCIL COUPLING
    // ====================================================================

    // Warp ballot: aggregate spikes across 8 neurons in voxel
    unsigned int spike_ballot = __ballot_sync(warp_mask, my_spike);
    unsigned int voxel_spikes = spike_ballot & voxel_mask;
    int n_spikes_in_voxel = __popc(voxel_spikes);

    // Max intensity reduction across voxel's 8 neurons (all participate)
    float my_intensity = my_spike ? neuron_tau(k) : 0.0f;
    for (int off = THREADS_PER_VOXEL / 2; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(warp_mask, my_intensity, off);
        my_intensity = fmaxf(my_intensity, other);
    }
    float spike_intensity = my_intensity + (float)n_spikes_in_voxel * 0.1f;

    // Neuron 0: shared memory stencil deposit + spike event emission
    if (is_valid && has_atoms && k == 0 && n_spikes_in_voxel > 0 && spike_grid[v] == 0) {
        spike_grid[v] = REFRACTORY_STEPS;

        // Signal preservation (RAF spike)
        int spike_source = (n_nearby_excited > 0 && uv_signal > 0.3f) ? 1 : 2;
        update_signal_preservation(v, timestep, spike_source,
            voxel_hit_grid, last_uv_step, coupled_spike_grid,
            primary_residue_id, primary_residue_count,
            entry, residue_ids, n_atoms, residue_step_causal);

        // Coupling emission is phase-split into
        // nhs_build_coupling_from_spike_grid. That kernel writes one voxel per
        // thread in fixed neighbor order, avoiding f32 atomic order drift.

        // Spike event emission
        int _arom_type = -1, _arom_res = -1;
        float _wl = 0.0f, _vibe = 0.0f;
        if (n_nearby_excited > 0 && closest_excited_idx >= 0) {
            _arom_type = d_aromatic_type[closest_excited_idx];
            _wl = uv_wavelength_nm;
            _vibe = d_vibrational_energy[closest_excited_idx];
            for (int wi = 0; wi < entry.n_atoms && _arom_res < 0; wi++) {
                int ai = entry.atom_indices[wi];
                if (ai >= 0 && ai < n_atoms && d_atom_to_aromatic[ai] == closest_excited_idx) {
                    _arom_res = residue_ids[ai];
                }
            }
        }

        int spike_idx = atomicAdd(spike_count, 1);
        if (spike_idx < max_spikes) {
            capture_spike_event(
                spike_events[spike_idx], timestep, v, voxel_center,
                spike_intensity, entry, residue_ids,
                spike_source, _wl, _arom_type, _arom_res,
                water_density[v], _vibe, n_nearby_excited,
                fabsf(water_density[v] - water_density_prev[v]),
                d_protocol->current_phase_bits
            );
        }
    }

    // ====================================================================
    // COUPLING FLUSH
    // ====================================================================
    __syncthreads();  // Ensure all stencil writes are visible
    // No global f32 atomic writes here. See nhs_build_coupling_from_spike_grid.

    // ====================================================================
    // PHASE 3: EFP (neuron 0 only — same as original)
    // ====================================================================
    if (is_valid && has_atoms && k == 0) {
        float phi = 0.0f;
        int n_charged_nearby = 0;

        for (int wi = 0; wi < entry.n_atoms; wi++) {
            int ai = entry.atom_indices[wi];
            if (ai < 0 || ai >= n_atoms) continue;
            float q = charges[ai];
            if (fabsf(q) < 0.15f) continue;

            float3 ap = positions[ai];
            float ddx = voxel_center.x - ap.x;
            float ddy = voxel_center.y - ap.y;
            float ddz = voxel_center.z - ap.z;
            float dist = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);

            if (dist > 0.5f && dist < 8.0f) {
                float eps_r = fmaxf(4.0f * dist, 4.0f);
                phi += q / (eps_r * dist);
                n_charged_nearby++;
            }
        }

        float wd_change = fabsf(water_density[v] - water_density_prev[v]);
        float polar_water_signal = fabsf(phi) * wd_change * 40.0f;

        float phi_prev = efp_potential_prev[v];
        efp_potential[v] = phi;

        float flux = fabsf(phi - phi_prev);
        float polar_signal = flux * 150.0f + polar_water_signal;

        if (n_charged_nearby >= 1 && spike_grid_efp[v] == 0) {
            const float EFP_TAU = 0.5f;
            const float EFP_THRESHOLD = 0.15f;
            float efp_decay = expf(-dt / EFP_TAU);
            efp_lif_potential[v] = efp_decay * efp_lif_potential[v] + polar_signal;

            if (efp_lif_potential[v] > EFP_THRESHOLD) {
                spike_grid_efp[v] = REFRACTORY_STEPS;
                float polar_intensity = efp_lif_potential[v];
                efp_lif_potential[v] = LIF_RESET;

                // Signal preservation (multi-LIF EFP spike)
                update_signal_preservation(v, timestep, 3,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    entry, residue_ids, n_atoms, residue_step_causal);

                int si = atomicAdd(spike_count, 1);
                if (si < max_spikes) {
                    int polar_type = (phi > 0.0f) ? 5 : 6;
                    capture_spike_event(
                        spike_events[si], timestep, v, voxel_center,
                        polar_intensity, entry, residue_ids,
                        3, 0.0f, polar_type, -1,
                        water_density[v], flux, n_charged_nearby,
                        wd_change,
                        d_protocol->current_phase_bits
                    );
                }
            }
        }
        efp_potential_prev[v] = phi;
    }
}

// Deterministic RAF coupling builder.
//
// The previous multi-LIF path deposited per-spike neighbor weights through
// shared/global f32 atomicAdd. That made the next-step coupling field depend
// on warp/block scheduling. This phase-split kernel assigns exactly one writer
// to each destination voxel and scans its 26 source neighbors in fixed
// lexicographic order. New RAF spikes are identified by spike_grid ==
// REFRACTORY_STEPS; older refractory cells were decremented before the current
// multi-LIF spike decision, so they are not re-emitted.
extern "C" __global__ void nhs_build_coupling_from_spike_grid(
    const int* __restrict__ spike_grid,
    float* __restrict__ coupling_a,
    float* __restrict__ coupling_b,
    int grid_dim,
    const ProtocolState* __restrict__ d_protocol
) {
    if (!prism_protocol_observation_due(d_protocol)) return;

    // Exact128-cube production route (also any full8-divisible cube<=MAX_GRID_DIM):
    // preserve the original1D256 launch and all scalar/predicate/float operations.
    // Every other grid/launch profile retains the original body below verbatim.
    const bool shared_halo_route =
        grid_dim >= 8 && grid_dim <= MAX_GRID_DIM && (grid_dim % 8) == 0 &&
        blockDim.x == 256 && blockDim.y == 1 && blockDim.z == 1 &&
        gridDim.y == 1 && gridDim.z == 1 &&
        gridDim.x == (unsigned)((grid_dim / 8) * (grid_dim / 8) * (grid_dim / 4));
    if (shared_halo_route) {
        // Logical 10x10x6 halo with an 11-word physical X stride (2640 B).
        // This preserves the requested layout; four 8-lane rows per warp still
        // have shared-bank collisions. Coprimality alone does not remove them.
        constexpr int COUPLING_HALO_X_LOGICAL = 10;
        constexpr int COUPLING_HALO_X_STRIDE = 11;
        constexpr int COUPLING_HALO_Y = 10;
        constexpr int COUPLING_HALO_Z = 6;
        __shared__ int spike_halo[COUPLING_HALO_Z * COUPLING_HALO_Y * COUPLING_HALO_X_STRIDE];
        const int tiles_x = grid_dim / 8;
        const int tiles_y = grid_dim / 8;
        const int tile_x = (int)blockIdx.x % tiles_x;
        const int tile_y = ((int)blockIdx.x / tiles_x) % tiles_y;
        const int tile_z = (int)blockIdx.x / (tiles_x * tiles_y);
        for (int h = (int)threadIdx.x; h < 600; h += 256) {
            const int hx = h % 10;
            const int hy = (h / 10) % 10;
            const int hz = h / 100;
            const int sx = tile_x * 8 + hx - 1;
            const int sy = tile_y * 8 + hy - 1;
            const int sz = tile_z * 4 + hz - 1;
            // Map every logical voxel to padded storage; no padding is read.
            const int halo_idx = hz * (COUPLING_HALO_Y * COUPLING_HALO_X_STRIDE)
                               + hy * COUPLING_HALO_X_STRIDE + hx;
            spike_halo[halo_idx] = (sx >= 0 && sx < grid_dim && sy >= 0 && sy < grid_dim &&
                             sz >= 0 && sz < grid_dim)
                ? spike_grid[(sz * grid_dim + sy) * grid_dim + sx] : 0;
        }
        // ALL256threads participate, including any future edge-output threads.
        // No perthreadreturn may precede publication of the complete halo.
        __syncthreads();
        const int local_x = (int)threadIdx.x % 8;
        const int local_y = ((int)threadIdx.x / 8) % 8;
        const int local_z = (int)threadIdx.x / 64;
        const int x = tile_x * 8 + local_x;
        const int y = tile_y * 8 + local_y;
        const int z = tile_z * 4 + local_z;
        if (x >= grid_dim || y >= grid_dim || z >= grid_dim) return;
        const int v = (z * grid_dim + y) * grid_dim + x;
        float coupling = 0.0f;
        for (int dz = -1; dz <= 1; dz++) {
            int sz = z + dz;
            if (sz < 0 || sz >= grid_dim) continue;
            for (int dy = -1; dy <= 1; dy++) {
                int sy = y + dy;
                if (sy < 0 || sy >= grid_dim) continue;
                for (int dx = -1; dx <= 1; dx++) {
                    int sx = x + dx;
                    if (sx < 0 || sx >= grid_dim) continue;
                    if (dx == 0 && dy == 0 && dz == 0) continue;
    
                    int src = sz * grid_dim * grid_dim + sy * grid_dim + sx;
                    if (spike_halo[(local_z + dz + 1) * (COUPLING_HALO_Y * COUPLING_HALO_X_STRIDE) + (local_y + dy + 1) * COUPLING_HALO_X_STRIDE + (local_x + dx + 1)] != REFRACTORY_STEPS) continue;
    
                    int manhattan = abs(dx) + abs(dy) + abs(dz);
                    coupling += (manhattan == 1) ? 1.0f :
                                (manhattan == 2) ? 0.7f : 0.5f;
                }
            }
        }

        float* coupling_write = (d_protocol->coupling_phase == 0) ? coupling_b : coupling_a;
        coupling_write[v] = coupling;
        return;
    }

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = grid_dim * grid_dim * grid_dim;
    if (v >= total_voxels) return;

    int z = v / (grid_dim * grid_dim);
    int y = (v / grid_dim) % grid_dim;
    int x = v % grid_dim;

    float coupling = 0.0f;
    for (int dz = -1; dz <= 1; dz++) {
        int sz = z + dz;
        if (sz < 0 || sz >= grid_dim) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int sy = y + dy;
            if (sy < 0 || sy >= grid_dim) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int sx = x + dx;
                if (sx < 0 || sx >= grid_dim) continue;
                if (dx == 0 && dy == 0 && dz == 0) continue;

                int src = sz * grid_dim * grid_dim + sy * grid_dim + sx;
                if (spike_grid[src] != REFRACTORY_STEPS) continue;

                int manhattan = abs(dx) + abs(dy) + abs(dz);
                coupling += (manhattan == 1) ? 1.0f :
                            (manhattan == 2) ? 0.7f : 0.5f;
            }
        }
    }

    float* coupling_write = (d_protocol->coupling_phase == 0) ? coupling_b : coupling_a;
    coupling_write[v] = coupling;
}

// Kernel to initialize multi-neuron RAF state
extern "C" __global__ void init_multi_neuron(
    float* neuron_potential,
    float* neuron_threshold,
    float* neuron_mean,
    int*   neuron_refractory,
    float* coupling_a,
    float* coupling_b,
    int    total_voxels
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize neuron arrays (total_voxels * K_NEURONS elements)
    if (tid < total_voxels * K_NEURONS) {
        neuron_potential[tid] = 0.01f;   // small x seed for oscillator
        neuron_threshold[tid] = MULTI_LIF_THRESHOLD;
        neuron_mean[tid] = 0.0f;        // y = 0 initially
        neuron_refractory[tid] = 0;
    }
    // Initialize coupling fields (total_voxels elements)
    if (tid < total_voxels) {
        coupling_a[tid] = 0.0f;
        coupling_b[tid] = 0.0f;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// KCC: EVENT-DRIVEN RING BUFFER WRITE HELPER
// ════════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void kcc_record_event(
    int residue_slot,
    unsigned int current_step,
    float dx, float dy, float dz,
    unsigned short c_count,
    unsigned short flags,
    KCCEvent* __restrict__ kcc_ring,
    unsigned int* __restrict__ kcc_ring_head,
    unsigned int* __restrict__ residue_last_event_step
) {
    unsigned int last_step = residue_last_event_step[residue_slot];

    if (last_step != current_step) {
        // New event: advance ring head, write fresh record
        unsigned int head = kcc_ring_head[residue_slot];
        unsigned int next = (head + 1u) % KCC_RING_STEPS;
        int idx = residue_slot * KCC_RING_STEPS + next;

        kcc_ring[idx].dx    = dx;
        kcc_ring[idx].dy    = dy;
        kcc_ring[idx].dz    = dz;
        kcc_ring[idx].c     = c_count;
        kcc_ring[idx].flags = flags;
        kcc_ring[idx].step  = current_step;

        kcc_ring_head[residue_slot] = next;
        residue_last_event_step[residue_slot] = current_step;
    } else {
        // Same step: accumulate into existing event
        unsigned int head = kcc_ring_head[residue_slot];
        int idx = residue_slot * KCC_RING_STEPS + head;

        kcc_ring[idx].c     += c_count;
        kcc_ring[idx].dx    += dx;
        kcc_ring[idx].dy    += dy;
        kcc_ring[idx].dz    += dz;
        kcc_ring[idx].flags |= flags;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// KCC v2-full: PER-RESIDUE KINEMATIC-CAUSAL UPDATE KERNEL
// ════════════════════════════════════════════════════════════════════════════
// Launched once per outer step, after multi-LIF. One thread per residue.
// Reads CA position, computes per-step motion, updates streaming reductions.
// Writes to event-driven ring buffer ONLY when c_t > 0.
__device__ __forceinline__ void kcc_residue_update_body(
    // Atom positions (read CA positions via index mapping)
    const float3* __restrict__ positions,
    const int* __restrict__ residue_ca_idx,  // [n_residues] atom index of CA per residue
    int n_residues,
    // Streaming reduction buffers [n_residues]
    float* residue_prev_x,
    float* residue_prev_y,
    float* residue_prev_z,
    float* residue_sum_m,      // sum of per-step motion magnitude
    float* residue_sum_m2,     // sum of squared motion magnitude
    float* residue_sum_c,      // sum of per-step causal activity
    float* residue_sum_c2,     // sum of squared causal activity
    float* residue_sum_mc,     // sum of motion * causality (covariance term)
    float* residue_net_dx,     // net displacement x
    float* residue_net_dy,     // net displacement y
    float* residue_net_dz,     // net displacement z
    unsigned int* residue_count,              // total steps tracked
    unsigned int* residue_active_causal_steps, // steps with c_t > 0
    // Per-residue per-step causal counter (read then zero)
    int* residue_step_causal,  // [n_residues] accumulated during spike emission
    // Event-driven ring buffer [n_residues * KCC_RING_STEPS]
    KCCEvent* kcc_ring,
    unsigned int* kcc_ring_head,              // [n_residues] circular write pointer
    unsigned int* residue_last_event_step,    // [n_residues] step of last event (init 0xFFFFFFFF)
    // Simulation state
    unsigned int current_step,                // absolute simulation timestep
    unsigned short flags                      // supplied by declared clock/phase producer
) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= n_residues) return;

    // Read CA position
    int ca_idx = residue_ca_idx[rid];
    if (ca_idx < 0) return;  // no CA for this residue (e.g. water)
    float3 pos = positions[ca_idx];

    // Previous position
    float px = residue_prev_x[rid];
    float py = residue_prev_y[rid];
    float pz = residue_prev_z[rid];

    // Per-step displacement
    float dx = pos.x - px;
    float dy = pos.y - py;
    float dz = pos.z - pz;
    float m_t = sqrtf(dx*dx + dy*dy + dz*dz);

    // Read and zero per-step causal counter
    int c_raw = residue_step_causal[rid];
    float c_t = (float)c_raw;
    residue_step_causal[rid] = 0;

    // Update streaming reductions (ALWAYS, regardless of causal activity)
    residue_sum_m[rid]  += m_t;
    residue_sum_m2[rid] += m_t * m_t;
    residue_sum_c[rid]  += c_t;
    residue_sum_c2[rid] += c_t * c_t;
    residue_sum_mc[rid] += m_t * c_t;
    residue_net_dx[rid] += dx;
    residue_net_dy[rid] += dy;
    residue_net_dz[rid] += dz;
    residue_count[rid]  += 1u;
    if (c_t > 0.0f) residue_active_causal_steps[rid] += 1u;

    // Event-driven ring buffer: write ONLY when causal activity occurred
    if (c_raw > 0) {
        unsigned short c_count = (unsigned short)min(c_raw, 65535);

        kcc_record_event(rid, current_step, dx, dy, dz, c_count, flags,
                         kcc_ring, kcc_ring_head, residue_last_event_step);
    }

    // Update previous position for next step (ALWAYS)
    residue_prev_x[rid] = pos.x;
    residue_prev_y[rid] = pos.y;
    residue_prev_z[rid] = pos.z;
}

extern "C" __global__ void kcc_residue_update(
    // Atom positions (read CA positions via index mapping)
    const float3* __restrict__ positions,
    const int* __restrict__ residue_ca_idx,  // [n_residues] atom index of CA per residue
    int n_residues,
    // Streaming reduction buffers [n_residues]
    float* residue_prev_x,
    float* residue_prev_y,
    float* residue_prev_z,
    float* residue_sum_m,      // sum of per-step motion magnitude
    float* residue_sum_m2,     // sum of squared motion magnitude
    float* residue_sum_c,      // sum of per-step causal activity
    float* residue_sum_c2,     // sum of squared causal activity
    float* residue_sum_mc,     // sum of motion * causality (covariance term)
    float* residue_net_dx,     // net displacement x
    float* residue_net_dy,     // net displacement y
    float* residue_net_dz,     // net displacement z
    unsigned int* residue_count,              // total steps tracked
    unsigned int* residue_active_causal_steps, // steps with c_t > 0
    // Per-residue per-step causal counter (read then zero)
    int* residue_step_causal,  // [n_residues] accumulated during spike emission
    // Event-driven ring buffer [n_residues * KCC_RING_STEPS]
    KCCEvent* kcc_ring,
    unsigned int* kcc_ring_head,              // [n_residues] circular write pointer
    unsigned int* residue_last_event_step,    // [n_residues] step of last event (init 0xFFFFFFFF)
    // Simulation state
    unsigned int current_step,                // absolute simulation timestep
    int current_phase                         // 0=cold_hold, 1=ramp, 2=warm_hold
) {
    const unsigned short flags = current_phase == 0 ? 1u : (current_phase == 2 ? 2u : 0u);
    kcc_residue_update_body(positions, residue_ca_idx, n_residues, residue_prev_x, residue_prev_y, residue_prev_z, residue_sum_m, residue_sum_m2, residue_sum_c, residue_sum_c2, residue_sum_mc, residue_net_dx, residue_net_dy, residue_net_dz, residue_count, residue_active_causal_steps, residue_step_causal, kcc_ring, kcc_ring_head, residue_last_event_step, current_step, flags);
}

extern "C" __global__ void kcc_residue_update_protocol(
    // Atom positions (read CA positions via index mapping)
    const float3* __restrict__ positions,
    const int* __restrict__ residue_ca_idx,  // [n_residues] atom index of CA per residue
    int n_residues,
    // Streaming reduction buffers [n_residues]
    float* residue_prev_x,
    float* residue_prev_y,
    float* residue_prev_z,
    float* residue_sum_m,      // sum of per-step motion magnitude
    float* residue_sum_m2,     // sum of squared motion magnitude
    float* residue_sum_c,      // sum of per-step causal activity
    float* residue_sum_c2,     // sum of squared causal activity
    float* residue_sum_mc,     // sum of motion * causality (covariance term)
    float* residue_net_dx,     // net displacement x
    float* residue_net_dy,     // net displacement y
    float* residue_net_dz,     // net displacement z
    unsigned int* residue_count,              // total steps tracked
    unsigned int* residue_active_causal_steps, // steps with c_t > 0
    // Per-residue per-step causal counter (read then zero)
    int* residue_step_causal,  // [n_residues] accumulated during spike emission
    // Event-driven ring buffer [n_residues * KCC_RING_STEPS]
    KCCEvent* kcc_ring,
    unsigned int* kcc_ring_head,              // [n_residues] circular write pointer
    unsigned int* residue_last_event_step,    // [n_residues] step of last event (init 0xFFFFFFFF)
    const ProtocolState* protocol,
    unsigned int* last_sample_step // per-residue update epoch; guards duplicate/reset
) {
    const int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= n_residues) return;
    const unsigned int step = protocol->current_step;
    if (step == 0 || step <= last_sample_step[rid]) { PRISM_PHYSICS_TRAP(); return; }
    // Match Director strict boundary comparisons and its explicit warm-tail mode.
    const long long s = step;
    const unsigned short phase_flag = s < protocol->cold_hold_end ? 1u
        : s < protocol->ramp_end ? 4u
        : s < protocol->warm_hold_end ? 2u
        : s < protocol->ramp_down_end ? 8u
        : (protocol->steering_flags & 0x10) ? 2u : 16u;
    // Bit15 marks endpoint-phase v2; no UV causality claim for interval counts.
    kcc_residue_update_body(positions, residue_ca_idx, n_residues, residue_prev_x, residue_prev_y, residue_prev_z, residue_sum_m, residue_sum_m2, residue_sum_c, residue_sum_c2, residue_sum_mc, residue_net_dx, residue_net_dy, residue_net_dz, residue_count, residue_active_causal_steps, residue_step_causal, kcc_ring, kcc_ring_head, residue_last_event_step, step, 0x8000u | phase_flag);
    last_sample_step[rid] = step;
}

// ════════════════════════════════════════════════════════════════════════════
// KCC v2-full: COMPUTE FINAL KCC METRICS FROM EVENT-DRIVEN RING BUFFER
// ════════════════════════════════════════════════════════════════════════════
// All temporal metrics use real Δt spacing from stored event timesteps.
extern "C" __global__ void kcc_compute_rich_descriptors(
    int n_residues,
    // Streaming reductions (read-only)
    const float* __restrict__ residue_sum_m,
    const float* __restrict__ residue_sum_m2,
    const float* __restrict__ residue_sum_c,
    const float* __restrict__ residue_sum_c2,
    const float* __restrict__ residue_sum_mc,
    const float* __restrict__ residue_net_dx,
    const float* __restrict__ residue_net_dy,
    const float* __restrict__ residue_net_dz,
    const unsigned int* __restrict__ residue_count,
    const unsigned int* __restrict__ residue_active_causal_steps,
    // Event-driven ring buffer (read-only)
    const KCCEvent* __restrict__ kcc_ring,
    const unsigned int* __restrict__ kcc_ring_head,
    // Output: per-residue KCC descriptors [n_residues]
    float* kcc_temporal_corr,
    float* kcc_direction_score,
    float* kcc_motion_efficiency,
    float* kcc_burst_motion_score,
    float* kcc_phase_shift_score,
    float* kcc_causal_lag,
    float* kcc_lag_corr_peak,
    float* kcc_local_cov_score
) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= n_residues) return;

    unsigned int N = residue_count[rid];
    if (N < 2) {
        kcc_temporal_corr[rid] = 0.0f;
        kcc_direction_score[rid] = 0.0f;
        kcc_motion_efficiency[rid] = 0.0f;
        kcc_burst_motion_score[rid] = 0.0f;
        kcc_phase_shift_score[rid] = 0.0f;
        kcc_causal_lag[rid] = 0.0f;
        kcc_lag_corr_peak[rid] = 0.0f;
        kcc_local_cov_score[rid] = 0.0f;
        return;
    }

    float fN = (float)N;

    // === Streaming temporal correlation (whole-run) ===
    float mean_m = residue_sum_m[rid] / fN;
    float mean_c = residue_sum_c[rid] / fN;
    float cov_mc = (residue_sum_mc[rid] / fN) - mean_m * mean_c;
    float var_m  = (residue_sum_m2[rid] / fN) - mean_m * mean_m;
    float var_c  = (residue_sum_c2[rid] / fN) - mean_c * mean_c;
    float sden   = sqrtf(fmaxf(var_m * var_c, 1e-12f));
    kcc_temporal_corr[rid] = cov_mc / sden;

    // === Direction score (whole-run) ===
    float ndx = residue_net_dx[rid], ndy = residue_net_dy[rid], ndz = residue_net_dz[rid];
    float net_disp = sqrtf(ndx*ndx + ndy*ndy + ndz*ndz);
    kcc_direction_score[rid] = net_disp / (residue_sum_m[rid] + 1e-6f);

    // === Motion efficiency (whole-run) ===
    kcc_motion_efficiency[rid] = 1.0f - expf(-residue_sum_m[rid] / (5.0f * fN));

    // === Read event-driven ring buffer (dense causal events only) ===
    unsigned int head = kcc_ring_head[rid];
    int base = rid * KCC_RING_STEPS;

    // Collect valid events in chronological order
    int n_events = 0;
    unsigned int ev_step[KCC_RING_STEPS];
    float ev_m[KCC_RING_STEPS];
    float ev_c[KCC_RING_STEPS];
    for (int i = 0; i < KCC_RING_STEPS; i++) {
        int slot = base + ((int)(head + 1 + i) % KCC_RING_STEPS);
        KCCEvent ev = kcc_ring[slot];
        if (ev.step == 0 && ev.c == 0) continue;  // empty slot
        ev_step[n_events] = ev.step;
        ev_m[n_events] = sqrtf(ev.dx*ev.dx + ev.dy*ev.dy + ev.dz*ev.dz);
        ev_c[n_events] = (float)ev.c;
        n_events++;
    }

    if (n_events < 4) {
        kcc_burst_motion_score[rid] = 0.0f;
        kcc_phase_shift_score[rid] = 0.0f;
        // Honest "not computable": insufficient events for cross-correlation.
        // NaN is the on-device contract; consumer (Rust + JSON emit) maps to None / null.
        kcc_causal_lag[rid] = nanf("");
        kcc_lag_corr_peak[rid] = nanf("");
        kcc_local_cov_score[rid] = 0.0f;
        return;
    }

    // === Burst: event density in real timestep space ===
    // Dense events (small Δt) = bursty; sparse events (large Δt) = sporadic
    float dt_vals[KCC_RING_STEPS];
    int n_dt = n_events - 1;
    for (int i = 0; i < n_dt; i++)
        dt_vals[i] = (float)(ev_step[i+1] - ev_step[i]);
    // Insertion sort for median
    for (int i = 1; i < n_dt; i++) {
        float key = dt_vals[i]; int j = i - 1;
        while (j >= 0 && dt_vals[j] > key) { dt_vals[j+1] = dt_vals[j]; j--; }
        dt_vals[j+1] = key;
    }
    float median_dt = n_dt > 0 ? dt_vals[n_dt / 2] : 1.0f;
    float sum_m_dense = 0.0f, sum_m_sparse = 0.0f;
    int n_dense = 0, n_sparse = 0;
    for (int i = 1; i < n_events; i++) {
        float dt = (float)(ev_step[i] - ev_step[i-1]);
        if (dt <= median_dt) { sum_m_dense += ev_m[i]; n_dense++; }
        else { sum_m_sparse += ev_m[i]; n_sparse++; }
    }
    float da = n_dense > 0 ? sum_m_dense / n_dense : 0.0f;
    float sa = n_sparse > 0 ? sum_m_sparse / n_sparse : da;
    kcc_burst_motion_score[rid] = da / (sa + 1e-6f);

    // === Phase shift: early vs late by real timestep range ===
    unsigned int step_min = ev_step[0];
    unsigned int step_max = ev_step[n_events - 1];
    unsigned int step_mid = (step_min + step_max) / 2;
    float early_m = 0.0f, late_m = 0.0f;
    int ne = 0, nl = 0;
    for (int i = 0; i < n_events; i++) {
        if (ev_step[i] <= step_mid) { early_m += ev_m[i]; ne++; }
        else { late_m += ev_m[i]; nl++; }
    }
    float ea = ne > 0 ? early_m / ne : 0.0f;
    float la = nl > 0 ? late_m / nl : 0.0f;
    kcc_phase_shift_score[rid] = (la - ea) / (ea + la + 1e-6f);

    // === Causal lag: cross-correlate c and m at real timestep offsets ===
    //
    // Honest cross-correlation contract (post producer-repair-causal-truthing):
    //   - j != i is required: self-matching is not a causal-lag signal.
    //     Allowing j == i made every residue's lag-search converge to a
    //     uniform self-correlation across all lags whose tolerance permitted
    //     j == i (lag in {-5, 0, 5}); strict `>` then locked in lag = -5
    //     because the loop walks lags from negative to positive and -5 was
    //     the first lag to qualify under the bd <= 5 self-match.
    //   - tie-break prefers smaller |lag|: an unbiased symmetric tie-break
    //     instead of "first lag wins".
    //   - n_valid_lags == 0 (no lag had >2 cross-paired events) writes NaN
    //     to both kcc_causal_lag and kcc_lag_corr_peak. NaN is the on-device
    //     contract for "not honestly computable"; consumer maps to None / null.
    float best_corr = -2.0f;       // distinct sentinel; valid corr ∈ [-1, 1]
    int   best_lag  = 0;
    int   n_valid_lags = 0;
    for (int lag = -50; lag <= 50; lag += 5) {
        float scm = 0.0f, scc = 0.0f, smm = 0.0f;
        int cnt = 0;
        for (int i = 0; i < n_events; i++) {
            int target = (int)ev_step[i] + lag;
            int bj = -1; int bd = 100000;
            for (int j = 0; j < n_events; j++) {
                if (j == i) continue;          // forbid self-match
                int d = abs((int)ev_step[j] - target);
                if (d < bd) { bd = d; bj = j; }
            }
            if (bj >= 0 && bd <= 5) {
                scm += ev_c[i] * ev_m[bj];
                scc += ev_c[i] * ev_c[i];
                smm += ev_m[bj] * ev_m[bj];
                cnt++;
            }
        }
        if (cnt > 2) {
            float d = sqrtf(fmaxf(scc * smm, 1e-12f));
            float corr = scm / d;
            n_valid_lags++;
            if (corr > best_corr ||
                (corr == best_corr && abs(lag) < abs(best_lag))) {
                best_corr = corr;
                best_lag  = lag;
            }
        }
    }
    if (n_valid_lags == 0) {
        kcc_causal_lag[rid]    = nanf("");
        kcc_lag_corr_peak[rid] = nanf("");
    } else {
        kcc_causal_lag[rid]    = (float)best_lag;
        kcc_lag_corr_peak[rid] = best_corr;
    }

    // === Local covariance: sliding windows in real timestep space ===
    unsigned int span = step_max - step_min;
    unsigned int W = span > 16 ? span / 4 : (span > 0 ? span : 4);
    if (W < 4) W = 4;
    float max_lcov = 0.0f;
    for (unsigned int ws = step_min; ws + W <= step_max; ws += W / 2) {
        unsigned int we = ws + W;
        float lsm = 0, lsc = 0, lsmc = 0, lsm2 = 0, lsc2 = 0;
        int wn = 0;
        for (int i = 0; i < n_events; i++) {
            if (ev_step[i] >= ws && ev_step[i] < we) {
                lsm += ev_m[i]; lsc += ev_c[i];
                lsmc += ev_m[i] * ev_c[i];
                lsm2 += ev_m[i] * ev_m[i];
                lsc2 += ev_c[i] * ev_c[i];
                wn++;
            }
        }
        if (wn >= 3) {
            float fw = (float)wn;
            float lcov = (lsmc/fw) - (lsm/fw)*(lsc/fw);
            float lvar = sqrtf(fmaxf((lsm2/fw-(lsm/fw)*(lsm/fw)) * (lsc2/fw-(lsc/fw)*(lsc/fw)), 1e-12f));
            float lc = lcov / lvar;
            if (lc > max_lcov) max_lcov = lc;
        }
    }
    kcc_local_cov_score[rid] = max_lcov;
}


// ═══════════════════════════════════════════════════════════════════════════
// GPU-ACCELERATED SPIKE DENSITY PEAK FINDING (KDE)
// ═══════════════════════════════════════════════════════════════════════════
//
// Computes burial-weighted Gaussian KDE on spike positions to find the
// density peak (hottest spot) within each detected pocket. This replaces
// the naive arithmetic centroid with a statistically robust estimate of
// the true pocket center.
//
// Architecture:
//   - One thread per 3D grid evaluation point (11³ = 1331 points per pocket)
//   - Shared memory caches spike positions + burial weights for the pocket
//   - Each thread evaluates the full KDE at its grid point
//   - Warp reduction finds the global maximum
//   - Final centroid = 70% density peak + 30% burial-weighted mean
//
// The burial weight (n_residues²) ensures that spikes deep inside the
// protein contribute more to the density field than surface spikes.
// This naturally pulls the peak toward the binding cavity interior.
//
// Complexity: O(n_grid_points × n_spikes_per_pocket)
// Typical: 1331 × 2000 = 2.6M kernel evaluations per pocket
// At ~1 TFLOP on RTX 5080: <1ms per pocket
// ═══════════════════════════════════════════════════════════════════════════

// Tile spikes through shared memory instead of caching the full pocket.
// RunPod RTX 5090 nodes currently expose a 48 KiB default per-block shared
// memory limit to the JIT. A 4096-entry SpikeKDE cache requires 64 KiB and
// causes CUDA_ERROR_INVALID_PTX at module load. A 2048-entry tile keeps the
// kernel under the portable 48 KiB limit while preserving every spike exactly.
#define KDE_TILE_SPIKES_SHARED 2048
#define KDE_GRID_DIM 11          // 11³ = 1331 grid points
#define KDE_GRID_TOTAL (KDE_GRID_DIM * KDE_GRID_DIM * KDE_GRID_DIM)

struct SpikeKDE {
    float x, y, z;      // position
    float weight;        // burial weight (n_residues²)
};

// Device function: evaluate Gaussian KDE at a point
__device__ __forceinline__ float kde_evaluate(
    float px, float py, float pz,
    const SpikeKDE* __restrict__ spikes,
    int n_spikes,
    float inv_2bw2  // 1 / (2 * bandwidth²)
) {
    float density = 0.0f;
    for (int i = 0; i < n_spikes; i++) {
        float dx = px - spikes[i].x;
        float dy = py - spikes[i].y;
        float dz = pz - spikes[i].z;
        float r2 = dx*dx + dy*dy + dz*dz;
        // Skip distant spikes (>3σ): exp(-9/2) ≈ 0.01
        if (r2 < 9.0f / inv_2bw2) {
            density += spikes[i].weight * expf(-r2 * inv_2bw2);
        }
    }
    return density;
}

// Kernel: find density peak for one pocket
// Launch: <<<ceil(KDE_GRID_TOTAL/256), 256>>> per pocket
// Inputs:
//   spike_positions: float3 array of spike xyz for this pocket
//   spike_n_residues: int array of nearby residue counts per spike
//   n_spikes: number of spikes in this pocket
//   centroid_in: [cx, cy, cz] initial centroid (burial-weighted mean from CPU)
//   search_radius: how far from centroid to search (typically 5.0A)
//   bandwidth: KDE bandwidth (typically 2.0A)
// Outputs:
//   peak_out: [px, py, pz, density] — position and density of the peak
extern "C" __global__ void kde_density_peak(
    const float* __restrict__ spike_x,
    const float* __restrict__ spike_y,
    const float* __restrict__ spike_z,
    const int*   __restrict__ spike_n_res,
    int          n_spikes,
    const float* __restrict__ centroid_in,  // [cx, cy, cz]
    float        search_radius,
    float        bandwidth,
    float*       peak_out                   // [px, py, pz, density]
) {
    // Shared memory for tiled spike cache and deterministic per-thread maxima.
    __shared__ SpikeKDE s_spikes[KDE_TILE_SPIKES_SHARED];
    __shared__ float s_thread_density[256];
    __shared__ float s_thread_x[256];
    __shared__ float s_thread_y[256];
    __shared__ float s_thread_z[256];
    __shared__ int s_thread_gid[256];

    int tid = threadIdx.x;

    // Host launch still uses ceil(1331/256) blocks. Keep the ABI intact but
    // let block 0 compute the whole deterministic reduction so peak_out is not
    // a cross-block last-writer race.
    if (blockIdx.x != 0) return;

    float cx = centroid_in[0];
    float cy = centroid_in[1];
    float cz = centroid_in[2];
    float grid_step = 2.0f * search_radius / (float)(KDE_GRID_DIM - 1);
    float inv_2bw2 = 1.0f / (2.0f * bandwidth * bandwidth);

    float local_best_density = 0.0f;
    float local_best_x = cx;
    float local_best_y = cy;
    float local_best_z = cz;
    int local_best_gid = KDE_GRID_TOTAL + 1;

    // Evaluate the complete 11^3 grid. All threads participate in each shared
    // spike tile load, so every __syncthreads() is reached by the full block.
    for (int base_gid = 0; base_gid < KDE_GRID_TOTAL; base_gid += blockDim.x) {
        int gid = base_gid + tid;
        bool active_grid_point = gid < KDE_GRID_TOTAL;
        float px = cx;
        float py = cy;
        float pz = cz;
        float density = 0.0f;

        if (active_grid_point) {
        int iz = gid / (KDE_GRID_DIM * KDE_GRID_DIM);
        int iy = (gid / KDE_GRID_DIM) % KDE_GRID_DIM;
        int ix = gid % KDE_GRID_DIM;

            px = cx - search_radius + ix * grid_step;
            py = cy - search_radius + iy * grid_step;
            pz = cz - search_radius + iz * grid_step;
        }

        for (int chunk = 0; chunk < n_spikes; chunk += KDE_TILE_SPIKES_SHARED) {
            int tile_count = min(n_spikes - chunk, KDE_TILE_SPIKES_SHARED);
            for (int i = tid; i < tile_count; i += blockDim.x) {
                int src = chunk + i;
                s_spikes[i].x = spike_x[src];
                s_spikes[i].y = spike_y[src];
                s_spikes[i].z = spike_z[src];
                // Burial weight: n_residues², minimum 1.0
                float nr = fmaxf((float)spike_n_res[src], 1.0f);
                s_spikes[i].weight = nr * nr;
            }
            __syncthreads();

            if (active_grid_point) {
                density += kde_evaluate(px, py, pz, s_spikes, tile_count, inv_2bw2);
            }
            __syncthreads();
        }

        if (active_grid_point &&
            (density > local_best_density ||
             (density == local_best_density && gid < local_best_gid))) {
            local_best_density = density;
            local_best_x = px;
            local_best_y = py;
            local_best_z = pz;
            local_best_gid = gid;
        }
    }

    s_thread_density[tid] = local_best_density;
    s_thread_x[tid] = local_best_x;
    s_thread_y[tid] = local_best_y;
    s_thread_z[tid] = local_best_z;
    s_thread_gid[tid] = local_best_gid;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int other = tid + stride;
            float od = s_thread_density[other];
            int og = s_thread_gid[other];
            if (od > s_thread_density[tid] ||
                (od == s_thread_density[tid] && og < s_thread_gid[tid])) {
                s_thread_density[tid] = od;
                s_thread_x[tid] = s_thread_x[other];
                s_thread_y[tid] = s_thread_y[other];
                s_thread_z[tid] = s_thread_z[other];
                s_thread_gid[tid] = og;
            }
        }
        __syncthreads();
    }

    // Thread 0 writes the final result: 70% peak + 30% centroid
    if (tid == 0) {
        peak_out[0] = 0.7f * s_thread_x[0] + 0.3f * cx;
        peak_out[1] = 0.7f * s_thread_y[0] + 0.3f * cy;
        peak_out[2] = 0.7f * s_thread_z[0] + 0.3f * cz;
        peak_out[3] = s_thread_density[0];
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// GPU-ACCELERATED BURIAL-WEIGHTED CENTROID
// ═══════════════════════════════════════════════════════════════════════════
//
// Computes weighted centroid of spike positions where weight = n_residues².
// Uses parallel reduction for O(log N) performance.
//
// Launch: <<<1, 256>>> per pocket
// ═══════════════════════════════════════════════════════════════════════════

extern "C" __global__ void burial_weighted_centroid(
    const float* __restrict__ spike_x,
    const float* __restrict__ spike_y,
    const float* __restrict__ spike_z,
    const int*   __restrict__ spike_n_res,
    int          n_spikes,
    float*       centroid_out    // [cx, cy, cz, total_weight]
) {
    __shared__ float s_wx[256];
    __shared__ float s_wy[256];
    __shared__ float s_wz[256];
    __shared__ float s_w[256];

    int tid = threadIdx.x;

    // Each thread accumulates over a strided range
    float wx = 0.0f, wy = 0.0f, wz = 0.0f, w = 0.0f;
    for (int i = tid; i < n_spikes; i += blockDim.x) {
        float nr = fmaxf((float)spike_n_res[i], 1.0f);
        float weight = nr * nr;
        wx += spike_x[i] * weight;
        wy += spike_y[i] * weight;
        wz += spike_z[i] * weight;
        w  += weight;
    }

    s_wx[tid] = wx;
    s_wy[tid] = wy;
    s_wz[tid] = wz;
    s_w[tid]  = w;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_wx[tid] += s_wx[tid + stride];
            s_wy[tid] += s_wy[tid + stride];
            s_wz[tid] += s_wz[tid + stride];
            s_w[tid]  += s_w[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float inv_w = (s_w[0] > 0.0f) ? 1.0f / s_w[0] : 0.0f;
        centroid_out[0] = s_wx[0] * inv_w;
        centroid_out[1] = s_wy[0] * inv_w;
        centroid_out[2] = s_wz[0] * inv_w;
        centroid_out[3] = s_w[0];
    }
}


// ============================================================================
// LADD: Local Atom Departure Detection (7-kernel pipeline)
// Merged from ladd-channel-20260331. Replaces the oscillator-based
// ladd_observation_step with departure-tracking + sorted scatter + depletion.
// ============================================================================

// Assign each heavy atom to its current voxel + detect departures.
// Also builds per-voxel departure count for O(1) indexed lookup.
extern "C" __global__ void ladd_update_voxels(
    const float3* __restrict__ positions,
    const unsigned char* __restrict__ atom_categories, // 255 = hydrogen (skip)
    int* atom_voxel_curr,
    int* atom_voxel_prev,
    int* departure_atoms,            // global departure buffer: atom indices
    int* departure_src_voxels,       // global departure buffer: source voxel
    int* departure_count,            // atomic counter (total departures)
    int* departure_count_per_voxel,  // [total_voxels]: departures from each voxel
    float grid_origin_x, float grid_origin_y, float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    int n_atoms,
    int max_departures
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_atoms) return;

    if (atom_categories[tid] == 255) {
        atom_voxel_curr[tid] = -1;
        return;
    }

    float3 pos = positions[tid];
    int ix = (int)((pos.x - grid_origin_x) / grid_spacing);
    int iy = (int)((pos.y - grid_origin_y) / grid_spacing);
    int iz = (int)((pos.z - grid_origin_z) / grid_spacing);

    int new_voxel;
    if (ix < 0 || ix >= grid_dim || iy < 0 || iy >= grid_dim ||
        iz < 0 || iz >= grid_dim) {
        new_voxel = -1;
    } else {
        new_voxel = ix + iy * grid_dim + iz * grid_dim * grid_dim;
    }

    int old_voxel = atom_voxel_prev[tid];
    atom_voxel_curr[tid] = new_voxel;

    if (old_voxel >= 0 && old_voxel != new_voxel) {
        int slot = atomicAdd(departure_count, 1);
        if (slot < max_departures) {
            departure_atoms[slot] = tid;
            departure_src_voxels[slot] = old_voxel;
        } else {
            // A truncated departure stream invalidates LADD/Ghost lineage.
            PRISM_PHYSICS_TRAP();
        }
        atomicAdd(&departure_count_per_voxel[old_voxel], 1);
    }
}

// Exclusive prefix sum on departure_count_per_voxel → departure_offsets.
extern "C" __global__ void ladd_build_departure_offsets(
    const int* __restrict__ departure_count_per_voxel,
    int* departure_offsets,   // [total_voxels + 1]
    int total_voxels
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int running = 0;
    for (int v = 0; v < total_voxels; v++) {
        departure_offsets[v] = running;
        running += departure_count_per_voxel[v];
    }
    departure_offsets[total_voxels] = running;
}

// Scatter departures into per-voxel sorted order using offsets.
extern "C" __global__ void ladd_scatter_departures(
    const int* __restrict__ departure_atoms,
    const int* __restrict__ departure_src_voxels,
    const int* __restrict__ departure_count,
    int max_departures,
    const int* __restrict__ departure_offsets,
    int* departure_local_slot,
    int* sorted_departure_atoms
) {
    // Native device count keeps this exact scatter capturable. The producer
    // traps on capacity exhaustion; reject a corrupt count before any read.
    const int n_departures = *departure_count;
    if (n_departures < 0 || n_departures > max_departures) {
        PRISM_PHYSICS_TRAP();
        return;
    }
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_departures) return;

    int src_voxel = departure_src_voxels[d];
    int atom = departure_atoms[d];
    int base = departure_offsets[src_voxel];
    int local = atomicAdd(&departure_local_slot[src_voxel], 1);
    sorted_departure_atoms[base + local] = atom;
}

// Reset per-voxel density grid (thread-per-voxel)
extern "C" __global__ void ladd_compute_density(
    const int* __restrict__ atom_voxel_curr,
    const unsigned char* __restrict__ atom_categories,
    float* ladd_voxel_density,
    int n_atoms,
    int total_voxels
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid < total_voxels) {
        ladd_voxel_density[vid] = 0.0f;
    }
}

// Accumulate heavy-atom density (thread-per-atom)
extern "C" __global__ void ladd_accumulate_density(
    const int* __restrict__ atom_voxel_curr,
    const unsigned char* __restrict__ atom_categories,
    float* ladd_voxel_density,
    int n_atoms
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_atoms) return;
    if (atom_categories[tid] == 255) return;
    int voxel = atom_voxel_curr[tid];
    if (voxel >= 0) {
        atomicAdd(&ladd_voxel_density[voxel], 1.0f);
    }
}

// Welford online mean during cold_hold phase
extern "C" __global__ void ladd_accumulate_reference(
    const float* __restrict__ ladd_voxel_density,
    float* ladd_density_ref,
    float* ladd_density_m2,
    int* ladd_ref_count,
    int total_voxels
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= total_voxels) return;

    float density = ladd_voxel_density[vid];
    int count = ladd_ref_count[vid] + 1;
    float delta = density - ladd_density_ref[vid];
    float new_mean = ladd_density_ref[vid] + delta / (float)count;
    float delta2 = density - new_mean;
    float new_m2 = ladd_density_m2[vid] + delta * delta2;

    ladd_density_ref[vid] = new_mean;
    ladd_density_m2[vid] = new_m2;
    ladd_ref_count[vid] = count;
}

// LADD constants
#define LADD_MIN_THRESHOLD 0.15f   // Minimum depletion fraction to fire
#define LADD_REFRACTORY    500     // Steps between LADD spikes per voxel
#define LADD_SOURCE        4       // spike_source value for LADD channel
#define COFIRE_SOURCE      5       // LADD+LIF co-fire (from HEAD)

// Detect depletion + analyze departures + emit LADD spikes.
// One thread per voxel. Uses sorted departure buffer for O(1) indexed lookup.
extern "C" __global__ void ladd_detect_and_emit(
    // Density data
    const float* __restrict__ ladd_voxel_density,
    const float* __restrict__ ladd_density_ref,
    const float* __restrict__ ladd_density_m2,
    const int* __restrict__ ladd_ref_count,
    int* ladd_spike_grid,          // refractory countdown
    // Sorted departure data (from ladd_scatter_departures)
    const int* __restrict__ sorted_departure_atoms,
    const int* __restrict__ departure_offsets,  // [total_voxels + 1]
    // Atom data for departure analysis
    const float3* __restrict__ positions,
    const float3* __restrict__ positions_prev,
    const unsigned char* __restrict__ atom_categories,
    const int* __restrict__ residue_ids,
    // Grid geometry
    float grid_origin_x, float grid_origin_y, float grid_origin_z,
    float grid_spacing,
    int grid_dim,
    // Spike output (shared with UV/LIF/EFP)
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    // Co-fire detection (from HEAD)
    const int* __restrict__ main_spike_grid,
    // Context
    int timestep,
    int total_voxels,
    const ProtocolState* d_protocol
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= total_voxels) return;

    // The captured cadence template reads the owning live physical clock.
    // Nonnegative explicit timestamps retain the existing host-dispatch ABI.
    // -1 is the named captured-device-clock mode, never an emitted timestamp.
    if (timestep == -1) {
        if (d_protocol == nullptr || d_protocol->current_step > 2147483647u) {
            PRISM_PHYSICS_TRAP();
            return;
        }
        timestep = (int)d_protocol->current_step;
    }
    if (timestep < 0) { PRISM_PHYSICS_TRAP(); return; }

    // Decrement refractory
    if (ladd_spike_grid[vid] > 0) {
        ladd_spike_grid[vid]--;
        return;
    }

    int ref_count = ladd_ref_count[vid];
    if (ref_count < 10) return;

    // Branch's departure-based detection body (replaces HEAD's oscillator)
    float ref_density = ladd_density_ref[vid];
    float curr_density = ladd_voxel_density[vid];
    if (ref_density < 0.5f) return;

    float depletion = (ref_density - curr_density) / ref_density;

    float variance = ladd_density_m2[vid] / fmaxf((float)(ref_count - 1), 1.0f);
    float sigma = sqrtf(variance);
    float threshold = fmaxf(3.0f * sigma / fmaxf(ref_density, 0.001f), LADD_MIN_THRESHOLD);

    if (depletion <= threshold) return;

    // ═══ DEPLETION DETECTED — direct-indexed departure lookup (O(D_local)) ═══

    int dep_start = departure_offsets[vid];
    int dep_end = departure_offsets[vid + 1];
    int n_departed = dep_end - dep_start;
    if (n_departed < 1) return;

    int departed_categories[8] = {0};
    float3 disp_sum = make_float3(0, 0, 0);
    float sum_disp_mag = 0.0f;
    float max_disp_sq = 0.0f;
    int max_disp_residue = -1;
    int departed_residues[8];
    int n_departed_residues = 0;

    for (int d = dep_start; d < dep_end; d++) {
        int atom = sorted_departure_atoms[d];

        unsigned char cat = atom_categories[atom];
        if (cat < 8) departed_categories[cat]++;

        float3 disp = make_float3(
            positions[atom].x - positions_prev[atom].x,
            positions[atom].y - positions_prev[atom].y,
            positions[atom].z - positions_prev[atom].z
        );
        disp_sum.x += disp.x;
        disp_sum.y += disp.y;
        disp_sum.z += disp.z;
        float d2 = disp.x*disp.x + disp.y*disp.y + disp.z*disp.z;
        sum_disp_mag += sqrtf(d2);
        int rid = residue_ids[atom];
        if (d2 > max_disp_sq && rid >= 0) {
            max_disp_sq = d2;
            max_disp_residue = rid;
        }

        if (rid < 0) continue;
        bool found = false;
        for (int r = 0; r < n_departed_residues; r++) {
            if (departed_residues[r] == rid) { found = true; break; }
        }
        if (!found && n_departed_residues < 8) {
            departed_residues[n_departed_residues++] = rid;
        }
    }

    // Displacement coherence: |Σdi| / Σ|di|
    float vec_mag = sqrtf(disp_sum.x*disp_sum.x + disp_sum.y*disp_sum.y + disp_sum.z*disp_sum.z);
    float coherence = (sum_disp_mag > 0.001f) ? (vec_mag / sum_disp_mag) : 0.0f;

    // Count adjacent depleted voxels (3x3x3 neighborhood)
    int vz = vid / (grid_dim * grid_dim);
    int vy = (vid / grid_dim) % grid_dim;
    int vx = vid % grid_dim;
    int n_adjacent_depleted = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = vx + dx, ny = vy + dy, nz = vz + dz;
                if (nx < 0 || nx >= grid_dim || ny < 0 || ny >= grid_dim ||
                    nz < 0 || nz >= grid_dim) continue;
                int nid = nx + ny * grid_dim + nz * grid_dim * grid_dim;
                float nref = ladd_density_ref[nid];
                if (nref > 0.5f) {
                    float ndep = (nref - ladd_voxel_density[nid]) / nref;
                    if (ndep > LADD_MIN_THRESHOLD) n_adjacent_depleted++;
                }
            }
        }
    }

    // ═══ EMIT LADD SPIKE (with co-fire detection from HEAD) ═══
    float3 voxel_center = make_float3(
        grid_origin_x + ((float)vx + 0.5f) * grid_spacing,
        grid_origin_y + ((float)vy + 0.5f) * grid_spacing,
        grid_origin_z + ((float)vz + 0.5f) * grid_spacing
    );

    // Co-fire detection: if LIF also spiked recently at this voxel, mark as COFIRE
    int source = LADD_SOURCE;
    if (main_spike_grid[vid] > 0 && main_spike_grid[vid] < LADD_REFRACTORY) {
        source = COFIRE_SOURCE;
    }

    int spike_idx = atomicAdd(spike_count, 1);
    if (spike_idx < max_spikes) {
        SpikeEvent event;
        event.timestep = timestep;
        event.phase_bits = d_protocol->current_phase_bits;
        event.voxel_idx = vid;
        event.position = voxel_center;
        event.intensity = depletion;
        event.spike_source = source;        // 4=LADD or 5=COFIRE
        event.wavelength_nm = 0.0f;
        event.aromatic_type = -1;
        event.aromatic_residue_id = max_disp_residue;
        event.water_density = curr_density;
        event.vibrational_energy = coherence;
        event.n_nearby_excited = n_adjacent_depleted;
        event.wd_change = (float)n_departed;

        event.n_residues = n_departed_residues;
        for (int r = 0; r < 8; r++) {
            event.nearby_residues[r] = (r < n_departed_residues) ? departed_residues[r] : -1;
        }

        spike_events[spike_idx] = event;
    }

    ladd_spike_grid[vid] = LADD_REFRACTORY;
}

// ═══════════════════════════════════════════════════════════════════════
// NMA-BIASED PERTURBATION KERNEL (from HEAD — preserved across LADD merge)
// Applies eigenvalue-scaled forces along normal mode directions to CA atoms.
// Called AFTER the main fused step, during warm_hold phase ONLY.
// ═══════════════════════════════════════════════════════════════════════

// The original modal force arithmetic, shared by the compatibility kernel and
// the physical force-phase composition below. No modal amplitude/cap changes.
__device__ inline float3 prism_nma_modal_force_v1(
    const float* displacements,const float* scales,int residues,int residue,
    int mode,float amplitude,float force_cap) {
    const int base=mode*residues*3+residue*3;
    float fx=amplitude*scales[mode]*displacements[base];
    float fy=amplitude*scales[mode]*displacements[base+1];
    float fz=amplitude*scales[mode]*displacements[base+2];
    const float magnitude=sqrtf(fx*fx+fy*fy+fz*fz);
    if(magnitude>force_cap){const float ratio=force_cap/magnitude;fx*=ratio;fy*=ratio;fz*=ratio;}
    return make_float3(fx,fy,fz);
}

// Preserve the existing scan/focus and literal 100-step cosine formulas while
// using the native owning protocol clock/boundaries instead of a frozen host step.
__device__ inline void prism_nma_schedule_at_step_v1(const ProtocolState* ps,int modes,
    float amplification,float scan_fraction,int step,int* mode,float* amplitude) {
    *mode=-1;*amplitude=0.0f;
    if(step<ps->ramp_end||step>=ps->warm_hold_end)return;
    const int into=step-ps->ramp_end;
    const int scan_steps=(int)((float)(ps->warm_hold_end-ps->ramp_end)*scan_fraction);
    const int per_mode=scan_steps/max(modes,1);
    *mode=into<scan_steps?min(into/max(per_mode,1),modes-1):0;
    const int begin=ps->ramp_end+(*mode)*per_mode;
    const int end=begin+per_mode;
    const int ramp=100;
    const float pi=3.14159265358979323846f;
    const float gate=step<begin+ramp?
        0.5f*(1.0f-cosf((float)(step-begin)*pi/(float)ramp)):
        (step>end-ramp?0.5f*(1.0f+cosf((float)(step-(end-ramp))*pi/(float)ramp)):1.0f);
    *amplitude=amplification*gate;
}

__device__ inline void prism_nma_schedule_v1(const ProtocolState* ps,int modes,
    float amplification,float scan_fraction,int* mode,float* amplitude) {
    prism_nma_schedule_at_step_v1(ps,modes,amplification,scan_fraction,(int)ps->current_step,mode,amplitude);
}

// state=[composed_steps,drained_steps,phase0_epoch,phase1_epoch]. Validation is
// serial and ordered before composition, so a full unread journal cannot race
// with parallel record writers. Optional journal capacity zero is unbound.
// Shared arithmetic and chronology are factored once for both native routes.
// The fusion entry has a single CTA; its barriers replace kernel boundaries.
constexpr int PRISM_NMA_FUSED_CTA_RESIDUE_CAPACITY=512;
__device__ inline bool prism_nma_phase_identity_valid_v1(
    const ProtocolState* ps,const unsigned long long* state,unsigned int capacity,
    unsigned int window_steps,unsigned int phase) {
    const unsigned long long step=ps->current_step;
    if(phase==3u||phase==4u) {
        return state[0]+1ull==step&&state[2]==state[0]&&state[3]==state[0]&&
            (!capacity||(state[0]<capacity&&state[1]<=state[0]&&window_steps>0&&state[0]-state[1]<window_steps));
    }
    bool valid=phase<=2;
    if(phase==0)valid=valid&&state[0]+1ull==step&&state[2]==state[0]&&state[3]==state[0];
    else if(phase==1)valid=valid&&state[0]+1ull==step&&state[2]==step&&state[3]==state[0];
    else valid=valid&&state[0]==step&&state[2]==step&&state[3]==step;
    if(capacity&&phase==0)valid=valid&&state[0]<capacity&&state[1]<=state[0]&&
        window_steps>0&&state[0]-state[1]<window_steps;
    return valid;
}
__device__ inline void prism_nma_report_phase_identity_v1(
    const ProtocolState* ps,const unsigned long long* state,unsigned int capacity,
    unsigned int window_steps,unsigned int phase) {
    const unsigned long long step=ps->current_step;
    printf("PRISM_NMA_PHASE_IDENTITY phase=%u step=%llu completed=%llu drained=%llu f0=%llu f1=%llu capacity=%u window=%u\n",
        phase,step,state[0],state[1],state[2],state[3],capacity,window_steps);
}

extern "C" __global__ void nhs_nma_validate_phase_v1(
    const ProtocolState* ps,const unsigned long long* state,unsigned int capacity,
    unsigned int window_steps,unsigned int phase) {
    if(threadIdx.x||blockIdx.x)return;
    if(!prism_nma_phase_identity_valid_v1(ps,state,capacity,window_steps,phase)) {
        prism_nma_report_phase_identity_v1(ps,state,capacity,window_steps,phase);
        PRISM_PHYSICS_TRAP();
    }
}

// Potential convention is declared, rather than inferred as a new restraint:
// U_NMA=-F_mode dot (x-x_reference), with immutable prepared-reference zero.
// Both physical kicks use the SAME force vector selected at phase0. Only its
// fixed-coordinate parameter change contributes control work. Phase2 is a
// read-only endpoint evaluation for an explicitly imposed coordinate intervention.
template<bool Compensated=false>
__device__ inline void prism_nma_compose_residue_v1(int r,
    const float3* x,const float3* reference,float3* force,double* pe,double* work,
    const float* displacements,const float* scales,const int* ca_indices,
    int residues,int modes,const float* controls,float force_cap,
    const ProtocolState* ps,unsigned int phase,float3* current_force,double* rows,
    const BiprismCoordinateOwnerFfi* coordinate_owner=nullptr,uint32_t* precision_fault=nullptr) {
    const int atom=ca_indices[r];
    if constexpr(Compensated) {
        if(atom<0 || (uint32_t)atom>=coordinate_owner->atom_count) {
            bp_native_coordinate_fault(precision_fault);return;
        }
    }
    const auto point=BpNativeGeometry<Compensated>::load(x,atom,coordinate_owner,precision_fault);
    const float3 origin=reference[atom]; // Immutable native reference, not MeshReference.
    if constexpr(Compensated) {
        if(!isfinite(point.x)||!isfinite(point.y)||!isfinite(point.z)||
           !isfinite(origin.x)||!isfinite(origin.y)||!isfinite(origin.z)) {
            bp_native_coordinate_fault(precision_fault);return;
        }
    }
    const double dx=(double)point.x-origin.x,dy=(double)point.y-origin.y,dz=(double)point.z-origin.z;
    float3 modal=current_force[r];double* row=rows+12ull*(unsigned long long)r;
    if(phase==0) {
        const double old_u=-((double)modal.x*dx+(double)modal.y*dy+(double)modal.z*dz);
        int mode;float amplitude;prism_nma_schedule_v1(ps,modes,controls[0],controls[1],&mode,&amplitude);
        modal=mode<0?make_float3(0,0,0):prism_nma_modal_force_v1(displacements,scales,residues,r,mode,amplitude,force_cap);
        const double new_u=-((double)modal.x*dx+(double)modal.y*dy+(double)modal.z*dz);
        const double parameter_work=new_u-old_u;
        if(!isfinite(new_u)||!isfinite(parameter_work)||!isfinite(amplitude)){PRISM_PHYSICS_TRAP();return;}
        current_force[r]=modal;work[atom]+=parameter_work;
        row[0]=modal.x;row[1]=modal.y;row[2]=modal.z;row[6]=new_u;
        row[8]=parameter_work;row[9]=mode;row[10]=amplitude;row[11]=(double)ps->current_step;
    }
    const double potential=-((double)modal.x*dx+(double)modal.y*dy+(double)modal.z*dz);
    force[atom].x+=modal.x;force[atom].y+=modal.y;force[atom].z+=modal.z;pe[atom]+=potential;
    if(phase==1){row[3]=modal.x;row[4]=modal.y;row[5]=modal.z;row[7]=potential;}
}

template<bool Compensated>
__device__ __forceinline__ void bp_native_nhs_nma_compose_force_v1_body(
    const float3* x,const float3* reference,float3* force,double* pe,double* work,
    const float* displacements,const float* scales,const int* ca_indices,
    int residues,int modes,const float* controls,float force_cap,
    const ProtocolState* ps,unsigned int phase,float3* current_force,double* rows,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault) {
    if constexpr(Compensated) {
        const bool valid=bp_native_coordinate_descriptor_aligned(coordinate_owner) && bp_native_coordinate_owner_valid(coordinate_owner,
            reinterpret_cast<const float*>(x),(int)coordinate_owner->atom_count,
            BP_COORD_CURRENT_INITIALIZED,precision_fault) && phase<=2u;
        if(!valid){bp_native_coordinate_fault(precision_fault);return;}
    }

    const int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=residues)return;
    prism_nma_compose_residue_v1<Compensated>(r,x,reference,force,pe,work,displacements,scales,ca_indices,residues,modes,controls,force_cap,ps,phase,current_force,rows,coordinate_owner,precision_fault);
}

extern "C" __global__ void nhs_nma_compose_force_v1(
    const float3* x,const float3* reference,float3* force,double* pe,double* work,
    const float* displacements,const float* scales,const int* ca_indices,
    int residues,int modes,const float* controls,float force_cap,
    const ProtocolState* ps,unsigned int phase,float3* current_force,double* rows) {
    bp_native_nhs_nma_compose_force_v1_body<false>(x, reference, force, pe, work, displacements, scales, ca_indices, residues, modes, controls, force_cap, ps, phase, current_force, rows,nullptr,nullptr);
}

extern "C" __global__ void nhs_nma_compose_force_compensated_v1(
    const float3* x,const float3* reference,float3* force,double* pe,double* work,
    const float* displacements,const float* scales,const int* ca_indices,
    int residues,int modes,const float* controls,float force_cap,
    const ProtocolState* ps,unsigned int phase,float3* current_force,double* rows,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault) {
    bp_native_nhs_nma_compose_force_v1_body<true>(x, reference, force, pe, work, displacements, scales, ca_indices, residues, modes, controls, force_cap, ps, phase, current_force, rows,coordinate_owner,precision_fault);
}

__device__ inline void prism_nma_commit_phase_cta_v1(
    const ProtocolState* ps,unsigned long long* state,unsigned int phase,
    const double* rows,double* window,unsigned int window_steps,int residues,double* summary) {
    const unsigned int tid=threadIdx.x;
    const unsigned long long step=ps->current_step;
    if(phase==0){if(tid==0)state[2]=step;return;}
    if(phase!=1){if(tid==0){PRISM_PHYSICS_TRAP();}return;}

    // A single CTA validates the full immutable source population before any
    // journal store. Only invalid, unsealed scratch-write timing differs from
    // the old serial loop; accepted rows and publication identities do not.
    __shared__ unsigned int invalid;
    if(tid==0)invalid=0;
    __syncthreads();
    const unsigned long long count=residues>0?(unsigned long long)residues:0ull;
    const unsigned long long fields=count*12ull;
    for(unsigned long long field=tid;field<fields;field+=blockDim.x) {
        const unsigned int k=(unsigned int)(field%12ull);
        const double value=rows[field];
        if(!isfinite(value) || (k==11 && value!=(double)step) ||
                (k<3 && value!=rows[field+3ull]))atomicOr(&invalid,1u);
    }
    __syncthreads();
    if(invalid){if(tid==0){PRISM_PHYSICS_TRAP();}return;}

    // Loads and copies are coalesced; arithmetic remains in the original
    // residue order on thread zero. No floating-point tree/atomic reduction.
    constexpr unsigned int TILE_RESIDUES=32;
    __shared__ double tile[TILE_RESIDUES*12];
    double parameter_work=0.0,potential_n=0.0,potential_np1=0.0;bool nonzero=false;
    const unsigned long long journal_base=(window && count>0)
        ? (state[0]%window_steps)*count*12ull:0ull;
    for(unsigned long long base=0;base<count;base+=TILE_RESIDUES) {
        const unsigned long long left=count-base;
        const unsigned int tile_residues=(unsigned int)(left<TILE_RESIDUES?left:TILE_RESIDUES);
        const unsigned int tile_fields=tile_residues*12u;
        for(unsigned int field=tid;field<tile_fields;field+=blockDim.x)
            tile[field]=rows[base*12ull+field];
        __syncthreads();
        if(tid==0) {
            for(unsigned int r=0;r<tile_residues;++r) {
                const double* row=tile+r*12u;
                parameter_work=__dadd_rn(parameter_work,row[8]);
                potential_n=__dadd_rn(potential_n,row[6]);
                potential_np1=__dadd_rn(potential_np1,row[7]);
                nonzero=nonzero||row[0]!=0.0||row[1]!=0.0||row[2]!=0.0;
            }
        }
        if(window) {
            for(unsigned int field=tid;field<tile_fields;field+=blockDim.x)
                window[journal_base+base*12ull+field]=tile[field];
        }
        // Every store retires before tile reuse and before thread-zero seal.
        __syncthreads();
    }
    if(tid!=0)return;
    state[3]=step;state[0]=step;
    summary[0]+=parameter_work;summary[1]=parameter_work;summary[2]=potential_n;summary[3]=potential_np1;
    if(nonzero)summary[4]+=1.0;
}

extern "C" __global__ void nhs_nma_commit_phase_v1(
    const ProtocolState* ps,unsigned long long* state,unsigned int phase,
    const double* rows,double* window,unsigned int window_steps,int residues,double* summary) {
    if(blockIdx.x)return;
    prism_nma_commit_phase_cta_v1(ps,state,phase,rows,window,window_steps,residues,summary);
}

extern "C" __global__ void nhs_nma_commit_phase_compensated_v1(
    const ProtocolState* ps,unsigned long long* state,unsigned int phase,
    const double* rows,double* window,unsigned int window_steps,int residues,double* summary,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault) {
    if(blockIdx.x)return;
    const bool valid=bp_native_coordinate_descriptor_aligned(coordinate_owner) && bp_native_coordinate_owner_valid(coordinate_owner,
        coordinate_owner->current_hi,(int)coordinate_owner->atom_count,BP_COORD_CURRENT_INITIALIZED,precision_fault);
    if(!__syncthreads_and(valid)){if(threadIdx.x==0)bp_native_coordinate_fault(precision_fault);return;}

    prism_nma_commit_phase_cta_v1(ps,state,phase,rows,window,window_steps,residues,summary);
}

// One bounded CTA executes the SAME phase predicate, per-residue operator and
// ordered commit. Larger populations retain the general three-kernel route.
// All256 threads reach both entry barriers, including inactive residue lanes.
template<bool Compensated>
__device__ __forceinline__ void bp_native_nhs_nma_phase_fused_v1_body(
    const float3* x,const float3* reference,float3* force,double* pe,double* work,
    const float* displacements,const float* scales,const int* ca_indices,
    int residues,int modes,const float* controls,float force_cap,
    const ProtocolState* ps,unsigned int phase,float3* current_force,double* rows,
    unsigned long long* state,unsigned int capacity,double* window,
    unsigned int window_steps,double* summary,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault) {
    if constexpr(Compensated) {
        const bool valid=bp_native_coordinate_descriptor_aligned(coordinate_owner) && bp_native_coordinate_owner_valid(coordinate_owner,
            reinterpret_cast<const float*>(x),(int)coordinate_owner->atom_count,
            BP_COORD_CURRENT_INITIALIZED,precision_fault) && phase<=2u;
        if(!__syncthreads_and(valid)) {if(threadIdx.x==0)bp_native_coordinate_fault(precision_fault);return;}
    }

    const unsigned int tid=threadIdx.x;
    // This predicate is uniform for every thread in each CTA. No grid-wide
    // barrier or cooperative-residency assumption is used.
    if(gridDim.x!=1 || gridDim.y!=1 || gridDim.z!=1 ||
       blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 ||
       residues<=0 || residues>PRISM_NMA_FUSED_CTA_RESIDUE_CAPACITY) {
        if(tid==0){PRISM_PHYSICS_TRAP();}
        return;
    }
    __shared__ unsigned int identity_valid;
    if(tid==0) {
        identity_valid=prism_nma_phase_identity_valid_v1(ps,state,capacity,window_steps,phase)?1u:0u;
        if(!identity_valid)prism_nma_report_phase_identity_v1(ps,state,capacity,window_steps,phase);
    }
    __syncthreads();
    if(!identity_valid) {
        if(tid==0){PRISM_PHYSICS_TRAP();}
        return;
    }
    if(phase==3u||phase==4u) {
        // Historical ordinary NMA was a late force-buffer-only epilogue, not
        // a force in either kick. Phase3 means no original caller action;
        // phase4 reproduces that late action once after the whole inner group.
        // reference is the live velocity plane for these two explicit opcodes.
        for(int r=(int)tid;r<residues;r+=(int)blockDim.x) {
            int mode=-1;float amplitude=0.0f;
            if(phase==4u)prism_nma_schedule_at_step_v1(ps,modes,controls[0],controls[1],
                (int)ps->current_step-1,&mode,&amplitude);
            const float3 modal=mode<0?make_float3(0,0,0):
                prism_nma_modal_force_v1(displacements,scales,residues,r,mode,amplitude,10.0f);
            // For explicit historical opcodes3/4, the float argument in the
            // modern force-cap slot is original ordinary self.dt. The modal
            // cap remains the original literal10; modern opcodes unchanged.
            const int atom=ca_indices[r];
            float estimate=0.0f;
            if(mode>=0) {
                force[atom].x+=modal.x;force[atom].y+=modal.y;force[atom].z+=modal.z;
                const float3 v=reference[atom];
                const float y_product=__fmul_rn(modal.y,v.y);
                const float xy=__fmaf_rn(modal.x,v.x,y_product);
                const float xyz=__fmaf_rn(modal.z,v.z,xy);
                estimate=__fmul_rn(force_cap,xyz);
            }
            if(!isfinite(estimate)||!isfinite(amplitude)||!isfinite(modal.x)||!isfinite(modal.y)||!isfinite(modal.z)) {
                PRISM_PHYSICS_TRAP();return;
            }
            current_force[r]=modal;double* row=rows+12ull*(unsigned long long)r;
            row[0]=row[3]=(double)modal.x;row[1]=row[4]=(double)modal.y;row[2]=row[5]=(double)modal.z;
            row[6]=0.0;row[7]=0.0;row[8]=(double)estimate;
            row[9]=(double)mode;row[10]=(double)amplitude;row[11]=(double)ps->current_step;
        }
        __syncthreads();
        if(tid==0)state[2]=ps->current_step;
        __syncthreads();
        prism_nma_commit_phase_cta_v1(ps,state,1u,rows,window,window_steps,residues,summary);
        return;
    }
    // Unique CA indices are admitted by load_nma_modes before these immutable
    // arrays are uploaded. Strided residue ownership has no shared atom writes.
    for(int r=(int)tid;r<residues;r+=(int)blockDim.x)
        prism_nma_compose_residue_v1<Compensated>(r,x,reference,force,pe,work,displacements,scales,ca_indices,residues,modes,controls,force_cap,ps,phase,current_force,rows,coordinate_owner,precision_fault);
    // Replaces the original compose->commit kernel completion boundary.
    __syncthreads();
    if constexpr(Compensated) {
        if(!__syncthreads_and(atomicAdd(precision_fault,0u)==0u))return;
    }
    if(phase<2u)prism_nma_commit_phase_cta_v1(ps,state,phase,rows,window,window_steps,residues,summary);
    // Phase2 deliberately performs no phase publication, row/work/summary
    // mutation, modal selection or journal copy, matching the existing route.
}

extern "C" __global__ __launch_bounds__(256) void nhs_nma_phase_fused_v1(
    const float3* x,const float3* reference,float3* force,double* pe,double* work,
    const float* displacements,const float* scales,const int* ca_indices,
    int residues,int modes,const float* controls,float force_cap,
    const ProtocolState* ps,unsigned int phase,float3* current_force,double* rows,
    unsigned long long* state,unsigned int capacity,double* window,
    unsigned int window_steps,double* summary) {
    bp_native_nhs_nma_phase_fused_v1_body<false>(x, reference, force, pe, work, displacements, scales, ca_indices, residues, modes, controls, force_cap, ps, phase, current_force, rows, state, capacity, window, window_steps, summary,nullptr,nullptr);
}

extern "C" __global__ __launch_bounds__(256) void nhs_nma_phase_fused_compensated_v1(
    const float3* x,const float3* reference,float3* force,double* pe,double* work,
    const float* displacements,const float* scales,const int* ca_indices,
    int residues,int modes,const float* controls,float force_cap,
    const ProtocolState* ps,unsigned int phase,float3* current_force,double* rows,
    unsigned long long* state,unsigned int capacity,double* window,
    unsigned int window_steps,double* summary,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault) {
    bp_native_nhs_nma_phase_fused_v1_body<true>(x, reference, force, pe, work, displacements, scales, ca_indices, residues, modes, controls, force_cap, ps, phase, current_force, rows, state, capacity, window, window_steps, summary,coordinate_owner,precision_fault);
}

extern "C" __global__ void nma_perturbation_kernel(
    float* __restrict__ forces,
    const float* __restrict__ velocities,
    const float* __restrict__ nma_displacements,
    const float* __restrict__ nma_force_scales,
    const int* __restrict__ ca_indices,
    float* __restrict__ nma_work_terms,
    int n_residues,
    int active_mode,
    float amplitude,
    float dt,
    float force_cap
) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= n_residues) return;

    int atom_idx = ca_indices[rid];
    int disp_offset = active_mode * n_residues * 3 + rid * 3;
    float dx = nma_displacements[disp_offset + 0];
    float dy = nma_displacements[disp_offset + 1];
    float dz = nma_displacements[disp_offset + 2];

    float scale = nma_force_scales[active_mode];
    float fx = amplitude * scale * dx;
    float fy = amplitude * scale * dy;
    float fz = amplitude * scale * dz;

    float f_mag = sqrtf(fx*fx + fy*fy + fz*fz);
    if (f_mag > force_cap) {
        float clamp = force_cap / f_mag;
        fx *= clamp; fy *= clamp; fz *= clamp;
    }

    int f_offset = atom_idx * 3;
    // ca_indices is one C-alpha per residue, so this kernel has a unique
    // writer for each force row. Direct adds avoid f32 atomic order drift.
    forces[f_offset + 0] += fx;
    forces[f_offset + 1] += fy;
    forces[f_offset + 2] += fz;

    float vx = velocities[f_offset + 0];
    float vy = velocities[f_offset + 1];
    float vz = velocities[f_offset + 2];
    float work = (fx * vx + fy * vy + fz * vz) * dt;
    if (nma_work_terms != nullptr) {
        // Unique writer per (mode, residue) slot. The per-mode total can be
        // reduced later in fixed residue order without cross-thread atomics.
        nma_work_terms[active_mode * n_residues + rid] += work;
    }
}

// Diagnostic-only launch: reads actual native planes, writes no physical state.
// PE is refreshed only at FORCE_N / FORCE_NP1; phase identifies stale PE views.
template<bool Compensated>
__device__ __forceinline__ void bp_native_diagnostic_phase_energy_body(
    const float3* positions,const float3* positions_step_start,
    const float3* velocities,const float3* forces,const float* masses,
    const double* pe_components,const double* work_components,const double* work_window,
    const ProtocolState* protocol,int n_atoms,int phase,float cutoff,
    const float* charges,const LJParam* lj,
    const int* exclusion_list,const int* exclusion_offsets,
    const int* pairs14_list,const int* pairs14_offsets,int exact_exceptions,
    const int* excited,int n_aromatics,int lambda_slots,
    const int* atom_to_aromatic,const float* ground,const float* charge_means,
    const int* aromatic_types,const float* electronic_population,unsigned int source_id,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault
) {
    using G=BpNativeGeometry<Compensated>;using GReal=typename G::Real;
    if constexpr(Compensated) {
        if(gridDim.x!=1 || gridDim.y!=1 || gridDim.z!=1 ||
           blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1) {
            if(threadIdx.x==0)bp_native_coordinate_fault(precision_fault);return;
        }
        const bool valid=bp_native_coordinate_owner_valid(coordinate_owner,
            reinterpret_cast<const float*>(positions),n_atoms,
            BP_COORD_CURRENT_INITIALIZED|BP_COORD_START_INITIALIZED,precision_fault,
            reinterpret_cast<const float*>(positions_step_start));
        if(!__syncthreads_and(valid))return;
    }
    if(protocol->current_step>10u)return;
    __shared__ double sums[8][256];
    const unsigned lane=threadIdx.x;
    double pe=0,ke=0,work=0,force_power=0,cross_jump=0,cross_endpoint_delta=0,cross_count=0;
    unsigned excitation_count=0;
    for(int a=0;a<n_aromatics;++a)excitation_count+=(excited[a]!=0);
    const bool crossing_available=(phase==5 && lambda_slots==0);
    const float cut2=cutoff*cutoff;
    for(int i=(int)lane;i<n_atoms;i+=blockDim.x) {
        const auto v=velocities[i];const auto f=forces[i];
        pe+=pe_components[i];work+=work_components[i];
        ke+=0.5*(double)masses[i]*((double)v.x*v.x+(double)v.y*v.y+(double)v.z*v.z)/418.4;
        force_power+=(double)f.x*v.x+(double)f.y*v.y+(double)f.z*v.z;
        if(!crossing_available)continue;
        float qi_scale;
        const float qi=uv_charge_for_state(i,charges,ground,atom_to_aromatic,
            aromatic_types,exact_exceptions < 0 ? nullptr : charge_means,excited,electronic_population,qi_scale);
        for(int j=i+1;j<n_atoms;++j) {
            const auto d=G::difference(positions,i,j,coordinate_owner,
                BiprismCoordinatePlane::Current,precision_fault);
            const auto d0=G::difference(positions_step_start,i,j,coordinate_owner,
                BiprismCoordinatePlane::StepStart,precision_fault);
            const GReal x=d.x,y=d.y,z=d.z,x0=d0.x,y0=d0.y,z0=d0.z;
            const GReal r2=x*x+y*y+z*z,r20=x0*x0+y0*y0+z0*z0;
            if constexpr(Compensated) {
                if(!isfinite(r2)||!isfinite(r20)) {
                    bp_native_coordinate_fault(precision_fault);continue;
                }
            }
            const bool now=r2<cut2 && r2>=0.01f,was=r20<cut2 && r20>=0.01f;
            if(now==was || r2<0.01f || r20<0.01f)continue;
            if(lambda_csr_contains(i,j,exclusion_list,exclusion_offsets))continue;
            const bool is14=lambda_csr_contains(i,j,pairs14_list,pairs14_offsets);
            if(is14 && exact_exceptions > 0)continue;
            const float sigma=0.5f*(lj[i].sigma+lj[j].sigma);
            const float epsilon=sqrtf(lj[i].epsilon*lj[j].epsilon);
            float qj_scale;
            const float qj=uv_charge_for_state(j,charges,ground,atom_to_aromatic,
                aromatic_types,exact_exceptions < 0 ? nullptr : charge_means,excited,electronic_population,qj_scale);
            const float qq=qi*qj;
            const float slj=is14?FUDGE_LJ:1.0f,sqq=is14?FUDGE_QQ:1.0f;
            double values[3];
            const GReal squares[3]={(GReal)cut2,r20,r2};
            for(int k=0;k<3;++k) {
                const GReal inv=bp_native_rsqrt(squares[k]),sr=sigma*inv,sr2=sr*sr,sr6=sr2*sr2*sr2,sr12=sr6*sr6;
                values[k]=(double)slj*4.0*(double)epsilon*((double)sr12-(double)sr6)
                    +(double)sqq*(double)COULOMB_CONSTANT*(double)qq*(double)inv
                    - (exact_exceptions < 0 ? 0.0 : ordinary_cutoff_potential_offset_v1<GReal>(sigma,epsilon,qq,cut2,slj,sqq));
            }
            cross_jump+=(now?1.0:-1.0)*values[0];
            cross_endpoint_delta+=(now?values[2]:0.0)-(was?values[1]:0.0);
            cross_count+=1.0;
        }
    }
    sums[0][lane]=pe;sums[1][lane]=ke;sums[2][lane]=work;sums[3][lane]=force_power;
    sums[4][lane]=cross_jump;sums[5][lane]=cross_endpoint_delta;sums[6][lane]=cross_count;
    __syncthreads();
    if constexpr(Compensated) {
        if(!__syncthreads_and(atomicAdd(precision_fault,0u)==0u))return;
    }
    for(unsigned stride=128;stride;stride>>=1) {
        if(lane<stride)for(int k=0;k<7;++k)sums[k][lane]+=sums[k][lane+stride];
        __syncthreads();
    }
    if(lane==0)printf("PRISM_PHASE_ENERGY source=%u native_step=%u step=%u phase=%d potential_refreshed=%u pe=%.17g ke=%.17g work_components=%.17g work_window=%.17g force_dot_velocity=%.17g dt=%.9g cutoff=%.9g crossing_available=%u excited=%u lambda_slots=%d charge_authority=native_uv_excited_charge potential_convention=%s cutoff_pairs=%.0f cutoff_boundary_jump=%.17g cutoff_endpoint_delta=%.17g\n",
        source_id,protocol->current_step,protocol->current_step,phase,static_cast<unsigned>(phase==0||phase==5||phase==7||phase==30||phase==40),sums[0][0],sums[1][0],sums[2][0],*work_window,sums[3][0],
        (double)protocol->dt,(double)cutoff,(unsigned)crossing_available,excitation_count,lambda_slots,
        exact_exceptions < 0 ? "historical_may_unshifted_cutoff_v1" : "ordinary_continuous_cutoff_v1",sums[6][0],sums[4][0],sums[5][0]);
}

extern "C" __global__ void nhs_diagnostic_phase_energy_v1(
    const float3* positions,const float3* positions_step_start,
    const float3* velocities,const float3* forces,const float* masses,
    const double* pe_components,const double* work_components,const double* work_window,
    const ProtocolState* protocol,int n_atoms,int phase,float cutoff,
    const float* charges,const LJParam* lj,
    const int* exclusion_list,const int* exclusion_offsets,
    const int* pairs14_list,const int* pairs14_offsets,int exact_exceptions,
    const int* excited,int n_aromatics,int lambda_slots,
    const int* atom_to_aromatic,const float* ground,const float* charge_means,
    const int* aromatic_types,const float* electronic_population,unsigned int source_id
) {
    bp_native_diagnostic_phase_energy_body<false>(positions, positions_step_start, velocities, forces, masses, pe_components, work_components, work_window, protocol, n_atoms, phase, cutoff, charges, lj, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, exact_exceptions, excited, n_aromatics, lambda_slots, atom_to_aromatic, ground, charge_means, aromatic_types, electronic_population, source_id, nullptr, nullptr);
}

extern "C" __global__ void nhs_diagnostic_phase_energy_compensated_v1(
    const float3* positions,const float3* positions_step_start,
    const float3* velocities,const float3* forces,const float* masses,
    const double* pe_components,const double* work_components,const double* work_window,
    const ProtocolState* protocol,int n_atoms,int phase,float cutoff,
    const float* charges,const LJParam* lj,
    const int* exclusion_list,const int* exclusion_offsets,
    const int* pairs14_list,const int* pairs14_offsets,int exact_exceptions,
    const int* excited,int n_aromatics,int lambda_slots,
    const int* atom_to_aromatic,const float* ground,const float* charge_means,
    const int* aromatic_types,const float* electronic_population,unsigned int source_id,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault
) {
    bp_native_diagnostic_phase_energy_body<true>(positions, positions_step_start, velocities, forces, masses, pe_components, work_components, work_window, protocol, n_atoms, phase, cutoff, charges, lj, exclusion_list, exclusion_offsets, pairs14_list, pairs14_offsets, exact_exceptions, excited, n_aromatics, lambda_slots, atom_to_aromatic, ground, charge_means, aromatic_types, electronic_population, source_id, coordinate_owner, precision_fault);
}



// Serial admission runs before ANY coordinate/work/constraint mutation.
extern "C" __global__ void nhs_ca_intervention_validate(
    const ProtocolState* ps,const double* summary,unsigned int capacity,
    const unsigned long long* drained,unsigned int window_capacity) {
    if(threadIdx.x||blockIdx.x)return;
    const double count=summary[0];
    if(!isfinite(count)||count<0.0||count!=floor(count)) {PRISM_PHYSICS_TRAP();return;}
    if(capacity) {
        const unsigned long long produced=(unsigned long long)count;
        if(!drained||!window_capacity||produced>=capacity||*drained>produced||
            produced-*drained>=window_capacity||ps->current_step!=produced+1ull) {
            printf("PRISM_CA_INTERVENTION_ADMISSION step=%u produced=%llu drained=%llu capacity=%u window=%u\n",
                ps->current_step,produced,drained?*drained:0ull,capacity,window_capacity);
            PRISM_PHYSICS_TRAP();return;
        }
    }
}

// Legacy CA coordinate relaxation is an explicit non-Hamiltonian intervention.
// Its original dimensionless alpha law is retained verbatim. Snapshot the
// collocated Hamiltonian and preexisting work BEFORE any coordinate mutation.
extern "C" __global__ void nhs_ca_intervention_begin(
    float3* x, float3* x_before, const float3* v, const float* masses,
    const double* pe, const double* work, double* before,
    const float3* reference, const int* mask, int atoms, const ProtocolState* ps) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=atoms)return;
    const double vv=(double)v[i].x*v[i].x+(double)v[i].y*v[i].y+(double)v[i].z*v[i].z;
    before[3*i]=pe[i];before[3*i+1]=0.5*(double)masses[i]*vv/418.4;
    before[3*i+2]=work[i];x_before[i]=x[i];
    if(!isfinite(before[3*i])||!isfinite(before[3*i+1])||!isfinite(before[3*i+2])) {
        PRISM_PHYSICS_TRAP();return;
    }
    if(mask==nullptr||mask[i]!=1)return;
    float k=2.0f*(300.0f/fmaxf(ps->current_temperature,10.0f));
    k=fminf(k,50.0f);const float alpha=fminf(k*0.001f,0.1f);
    x[i].x+=alpha*(reference[i].x-x[i].x);
    x[i].y+=alpha*(reference[i].y-x[i].y);
    x[i].z+=alpha*(reference[i].z-x[i].z);
}

extern "C" __global__ void nhs_ca_intervention_begin_compensated_v1(
    float3* x, float3* x_before, const float3* v, const float* masses,
    const double* pe, const double* work, double* before,
    const float3* reference, const int* mask, int atoms, const ProtocolState* ps,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault) {
    if(!bp_native_coordinate_owner_valid(coordinate_owner,reinterpret_cast<const float*>(x),atoms,
        BP_COORD_CURRENT_INITIALIZED|BP_COORD_START_INITIALIZED,precision_fault,
        reinterpret_cast<const float*>(x_before)))return;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=atoms)return;
    const double vv=(double)v[i].x*v[i].x+(double)v[i].y*v[i].y+(double)v[i].z*v[i].z;
    before[3*i]=pe[i];before[3*i+1]=0.5*(double)masses[i]*vv/418.4;
    before[3*i+2]=work[i];
    if(!BpNativeGeometry<true>::snapshot(x_before,x,i,coordinate_owner,precision_fault))return;
    if(!isfinite(before[3*i])||!isfinite(before[3*i+1])||!isfinite(before[3*i+2])) {
        PRISM_PHYSICS_TRAP();return;
    }
    if(mask==nullptr||mask[i]!=1)return;
    float k=2.0f*(300.0f/fmaxf(ps->current_temperature,10.0f));
    k=fminf(k,50.0f);const float alpha=fminf(k*0.001f,0.1f);
    const auto current=bp_coord_load(*coordinate_owner,BiprismCoordinatePlane::Current,(uint32_t)i);
    // Original CA reference is immutable FP32; it is not the mesh-reference plane.
    const BiprismCoordinateExpansion3 reference_pair={{reference[i].x,0.0f},
        {reference[i].y,0.0f},{reference[i].z,0.0f}};
    const auto delta=bp_coord_difference(reference_pair,current);
    const BiprismCoordinateExpansion3 next={bp_coord_fma(current.x,(double)alpha,delta.x),
        bp_coord_fma(current.y,(double)alpha,delta.y),bp_coord_fma(current.z,(double)alpha,delta.z)};
    if(!bp_coord_store_current(*coordinate_owner,(uint32_t)i,next))bp_native_coordinate_fault(precision_fault);
}

// Read-only evaluation of the SAME active ASC potential at projected x. This
// does not fabricate a third force/kick phase or overwrite their event records.
extern "C" __global__ void nhs_ca_intervention_asc_endpoint(
    PrismAscDeviceV1* asc,const float3* x,float3* f,double* pe,int atoms) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=atoms||!asc)return;
    if(!prism_asc_phase_valid(asc,(uint32_t)atoms))return;
    if(!prism_asc_atom_selected(*asc,(uint32_t)i))return;
    const float xyz[3]={x[i].x,x[i].y,x[i].z};double force[3],potential;
    prism_asc_law(asc->active,xyz,force,&potential);
    if(!isfinite(potential)||!isfinite(force[0])||!isfinite(force[1])||!isfinite(force[2])) {
        PRISM_PHYSICS_TRAP();return;
    }
    f[i].x+=(float)force[0];f[i].y+=(float)force[1];f[i].z+=(float)force[2];pe[i]+=potential;
}

extern "C" __global__ void nhs_ca_intervention_asc_endpoint_compensated_v1(
    PrismAscDeviceV1* asc,const float3* x,float3* f,double* pe,int atoms,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault) {
    if(!bp_native_coordinate_owner_valid(coordinate_owner,reinterpret_cast<const float*>(x),atoms,
        BP_COORD_CURRENT_INITIALIZED,precision_fault))return;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=atoms||!asc)return;
    if(!prism_asc_phase_valid(asc,(uint32_t)atoms))return;
    if(!prism_asc_atom_selected(*asc,(uint32_t)i))return;
    const auto point=BpNativeGeometry<true>::load(x,i,coordinate_owner,precision_fault);
    const double xyz[3]={point.x,point.y,point.z};double force[3],potential;
    prism_asc_law_f64(asc->active,xyz,force,&potential);
    if(!isfinite(potential)||!isfinite(force[0])||!isfinite(force[1])||!isfinite(force[2])) {
        PRISM_PHYSICS_TRAP();return;
    }
    f[i].x+=(float)force[0];f[i].y+=(float)force[1];f[i].z+=(float)force[2];pe[i]+=potential;
}

// SHAKE/RATTLE may temporarily add kinetic bookkeeping during the projection.
// Replace ONLY that temporary increment with exact endpoint deltaH, including
// coordinate-dependent AMBER/lambda/PME/ASC potential and all constraint impulses.
extern "C" __global__ void nhs_ca_intervention_finish(
    const float3* v,const float* masses,const double* pe,const double* before,
    double* work,double* isolated,int atoms, double* window,const double* summary,unsigned int window_capacity) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=atoms)return;
    const double vv=(double)v[i].x*v[i].x+(double)v[i].y*v[i].y+(double)v[i].z*v[i].z;
    const double du=pe[i]-before[3*i];
    const double dk=0.5*(double)masses[i]*vv/418.4-before[3*i+1];
    if(!isfinite(du)||!isfinite(dk)) {PRISM_PHYSICS_TRAP();return;}
    isolated[3*i]=du;isolated[3*i+1]=dk;isolated[3*i+2]=du+dk;
    work[i]=before[3*i+2]+du+dk;
    if(!isfinite(work[i])){PRISM_PHYSICS_TRAP();return;}
    if(window) {
        const unsigned long long slot=((unsigned long long)summary[0])%window_capacity;
        double* row=window+(slot*(unsigned long long)atoms+(unsigned long long)i)*3ull;
        row[0]=du;row[1]=dk;row[2]=du+dk;
    }
}

// One deterministic reduction per physical step. The optional journal is bound
// to an explicit campaign horizon before capture; exhaustion traps before any
// evidence can claim coverage. Summary exists even for unjournaled legacy users.
extern "C" __global__ void nhs_ca_intervention_record(
    const double* isolated,int atoms,const ProtocolState* ps,double* summary,
    double* journal,unsigned int capacity,const PrismAscDeviceV1* asc) {
    if(blockIdx.x)return;
    const unsigned int tid=threadIdx.x;
    // Negative atom count is the explicit no-action record opcode. Its
    // absolute value is the complete same atom population; no capacity bits
    // or modern admissible-capacity domain are repurposed.
    const bool observation_only=atoms<0;
    if(atoms==(-2147483647-1)){if(tid==0){PRISM_PHYSICS_TRAP();}return;}
    if(observation_only)atoms=-atoms;
    constexpr unsigned int TILE_ATOMS=256;
    __shared__ double tile[TILE_ATOMS*3];
    double du=0.0,dk=0.0,dw=0.0;
    const unsigned long long count=atoms>0?(unsigned long long)atoms:0ull;
    for(unsigned long long base=0;base<count;base+=TILE_ATOMS) {
        const unsigned long long left=count-base;
        const unsigned int tile_atoms=(unsigned int)(left<TILE_ATOMS?left:TILE_ATOMS);
        const unsigned int fields=tile_atoms*3u;
        // Flat component addresses are contiguous across the CTA lanes.
        for(unsigned int field=tid;field<fields;field+=blockDim.x)
            tile[field]=isolated[base*3ull+field];
        __syncthreads();
        if(tid==0) {
            // Exactly the old atom-index addition order and f64 rounding.
            // No partial sums, reassociation or floating-point atomics.
            for(unsigned int i=0;i<tile_atoms;++i) {
                du=__dadd_rn(du,tile[3u*i]);
                dk=__dadd_rn(dk,tile[3u*i+1u]);
                dw=__dadd_rn(dw,tile[3u*i+2u]);
            }
        }
        // Every lane must wait before overwriting a consumed shared tile.
        __syncthreads();
    }
    if(tid!=0)return;
    if(!isfinite(du)||!isfinite(dk)||!isfinite(dw)){PRISM_PHYSICS_TRAP();return;}
    const unsigned int step=ps->current_step;
    if(journal) {
        const unsigned long long index=(unsigned long long)summary[0];
        if(index>=capacity){printf("PRISM_CA_WORK_JOURNAL_CAPACITY step=%u capacity=%u\n",step,capacity);PRISM_PHYSICS_TRAP();return;}
        double* row=journal+index*8ull;row[0]=(double)step;row[1]=du;row[2]=dk;row[3]=dw;
        const float k=fminf(2.0f*(300.0f/fmaxf(ps->current_temperature,10.0f)),50.0f);
        row[4]=(double)ps->current_temperature;row[5]=observation_only?0.0:(double)fminf(k*0.001f,0.1f);
        row[6]=(double)ps->current_phase_bits;
        // Current graph-bound control identity, not a new intervention decision.
        row[7]=asc?(double)asc->active.producer_epoch:-1.0;
    }
    summary[0]+=1.0;summary[1]=(double)step;
    summary[2]+=du;summary[3]+=dk;summary[4]+=dw;
    summary[5]=du;summary[6]=dk;summary[7]=dw;
}

// A separate native ledger for deterministic constraint impulses. These exact
// atom delta-K measurements are retained but are not bath/controller work.
// No residual, total-Hamiltonian correction or inferred heat is computed here.
extern "C" __global__ void nhs_conservative_constraint_begin_v1(
    unsigned long long* state,const ProtocolState* protocol,
    unsigned int horizon,unsigned int capacity) {
    if(blockIdx.x||threadIdx.x)return;
    const unsigned long long step=protocol->current_step;
    if(!step || state[2]!=0 || (capacity &&
        (step!=state[0]+1ull || step>horizon || state[0]<state[1] || state[0]-state[1]>=capacity))) {
        PRISM_PHYSICS_TRAP();return;
    }
    state[2]=step;
}
extern "C" __global__ void nhs_conservative_constraint_commit_v1(
    const double* pre,const double* shake,const double* post,
    unsigned long long* state,const ProtocolState* protocol,double* window,
    unsigned long long atoms,unsigned int capacity,unsigned int source,unsigned int diagnose) {
    const unsigned long long step=protocol->current_step;
    const unsigned int tid=threadIdx.x;
    if(blockIdx.x)return;
    if(state[2]!=step || !step || (capacity && (!window || step!=state[0]+1ull))) {
        PRISM_PHYSICS_TRAP();return;
    }
    for(unsigned long long atom=tid;atom<atoms;atom+=blockDim.x) {
        if(!isfinite(pre[atom])||!isfinite(shake[atom])||!isfinite(post[atom])) {PRISM_PHYSICS_TRAP();return;}
        if(capacity) {
            const unsigned long long base=((step-1ull)%capacity)*3ull*atoms;
            window[base+atom]=pre[atom];
            window[base+atoms+atom]=shake[atom];
            window[base+2ull*atoms+atom]=post[atom];
        }
    }
    // One block cooperatively copies the bounded row; seal only after every
    // atom store completes. No inter-block publication race.
    __syncthreads();
    if(tid==0) {
        if(diagnose && step<=10ull) {
            double a=0,b=0,c=0;for(unsigned long long i=0;i<atoms;++i){a+=pre[i];b+=shake[i];c+=post[i];}
            printf("PRISM_CONSERVATIVE_CONSTRAINT_IMPULSE source=%u native_step=%llu rattle_n_deltaK=%.17g shake_deltaK=%.17g rattle_np1_deltaK=%.17g accounting=deterministic_constraint_not_external_heat_or_control_work\n",source,step,a,b,c);
        }
        state[0]=step;state[2]=0;
    }
}


// Observer-only physical chronometry. Both nodes execute on the owning MD
// stream. They read the exact ProtocolState dt used by the physical kernels;
// they never write ProtocolState, coordinates, forces, controls or random state.
#include "prism_observer_physical_clock.cuh"
extern "C" __global__ void nhs_observer_physical_clock_begin(
    const ProtocolState* protocol,PrismObserverPhysicalClockStateV1* clock) {
    if(blockIdx.x!=0 || threadIdx.x!=0)return;
    if(!prism_observer_clock_begin_v1(clock,(uint64_t)protocol->current_step,protocol->dt)) {
        PRISM_PHYSICS_TRAP();
    }
}
extern "C" __global__ void nhs_observer_physical_clock_commit(
    const ProtocolState* protocol,PrismObserverPhysicalClockStateV1* clock,
    PrismObserverPhysicalTimeRecordV1* records) {
    if(blockIdx.x!=0 || threadIdx.x!=0)return;
    if(!prism_observer_clock_commit_v1(clock,records,(uint64_t)protocol->current_step,protocol->dt)) {
        PRISM_PHYSICS_TRAP();
    }
}

// Versioned Biprism same-emission native density/context exports.
#include "nhs_biprism_native_observation.cuh"
