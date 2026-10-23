// ═══════════════════════════════════════════════════════════════════════════════
// PRISM-TWIN v3.0 Gate 2 — Housekeeping Kernels
// ═══════════════════════════════════════════════════════════════════════════════
//
// GPU-side operations that replace CPU memcpy round-trips, enabling full
// CUDA Graph capture. All kernels read dynamic state from ProtocolState*.
//
// Kernels:
//   heartbeat_check       — NaN/divergence detection (samples every 32nd atom)
//   apply_ca_restraints   — harmonic CA position restraints (replaces CPU memcpy)
//   coupling_buffer_clear — zeroes the stale read-side coupling buffer based on phase
//   com_reduce            — warp-shuffle mass-weighted velocity reduction
//   com_correct           — broadcast COM velocity subtraction
// ═══════════════════════════════════════════════════════════════════════════════

#include "protocol_state.cuh"
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
// heartbeat_check — detect NaN or diverged coordinates
// ─────────────────────────────────────────────────────────────────────────────
//
// Samples every 32nd atom for efficiency. If any sampled coordinate is NaN
// or exceeds 1000 Angstroms, sets status_code in ProtocolState.
// Launch: <<<ceil(n_atoms/32/256), 256>>>

extern "C" __global__ void heartbeat_check(
    const float* __restrict__ positions,  // [n_atoms * 3]
    int n_atoms,
    ProtocolState* __restrict__ d_protocol
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int atom_idx = tid * 32;  // sample every 32nd atom
    if (atom_idx >= n_atoms) return;

    float x = positions[atom_idx * 3];
    float y = positions[atom_idx * 3 + 1];
    float z = positions[atom_idx * 3 + 2];

    if (isnan(x) || isnan(y) || isnan(z)) {
        d_protocol->status_code = 1;  // NaN detected
    } else if (fabsf(x) > 1000.0f || fabsf(y) > 1000.0f || fabsf(z) > 1000.0f) {
        d_protocol->status_code = 2;  // System diverged
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_ca_restraints — GPU-side harmonic position restraints
// ─────────────────────────────────────────────────────────────────────────────
//
// Replaces the CPU memcpy_dtoh → correct → memcpy_htod bottleneck.
// Formula matches fused_engine.rs exactly:
//   k = k_base * (300 / max(T, 10)), clamped to 50
//   alpha = min(k * 0.001, 0.1)
//   pos[i] += alpha * (ref[i] - pos[i])   for CA atoms only
//
// Launch: <<<ceil(n_atoms/256), 256>>>

extern "C" __global__ void apply_ca_restraints(
    float* __restrict__ positions,           // [n_atoms * 3] — modified in-place
    const float* __restrict__ ref_positions,  // [n_atoms * 3] — reference positions
    const int* __restrict__ ca_mask,          // [n_atoms] — 1 for CA, 0 otherwise
    int n_atoms,
    const ProtocolState* __restrict__ d_protocol
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_atoms) return;
    if (ca_mask[tid] != 1) return;

    float temp = d_protocol->current_temperature;
    float k = 2.0f * (300.0f / fmaxf(temp, 10.0f));
    k = fminf(k, 50.0f);
    float alpha = fminf(k * 0.001f, 0.1f);

    int base = tid * 3;
    positions[base]     += alpha * (ref_positions[base]     - positions[base]);
    positions[base + 1] += alpha * (ref_positions[base + 1] - positions[base + 1]);
    positions[base + 2] += alpha * (ref_positions[base + 2] - positions[base + 2]);
}

// ─────────────────────────────────────────────────────────────────────────────
// coupling_buffer_clear — zero the stale read-side coupling buffer
// ─────────────────────────────────────────────────────────────────────────────
//
// Reads coupling_phase from ProtocolState to determine which buffer was read
// during the just-finished step. Phase 0 read A / wrote B, so clear A. Phase 1
// read B / wrote A, so clear B. The newly written buffer must survive because
// the Director toggles coupling_phase at the next step and reads it then.
// Replaces the CPU-conditional memset_zeros.
//
// Launch: <<<ceil(total_voxels/256), 256>>>

extern "C" __global__ void coupling_buffer_clear(
    float* __restrict__ coupling_a,  // [total_voxels]
    float* __restrict__ coupling_b,  // [total_voxels]
    int total_voxels,
    const ProtocolState* __restrict__ d_protocol
) {
    if (!prism_protocol_observation_due(d_protocol)) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_voxels) return;

    // Director already toggled coupling_phase for THIS step.
    // Phase 0: read A, write B → clear stale A
    // Phase 1: read B, write A → clear stale B
    if (d_protocol->coupling_phase == 0) {
        coupling_a[tid] = 0.0f;
    } else {
        coupling_b[tid] = 0.0f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// COM velocity removal — two-kernel approach
// ─────────────────────────────────────────────────────────────────────────────
//
// Kernel 1: com_reduce — warp-shuffle reduction of mass-weighted velocities.
//   Each warp reduces its atoms, then one thread per warp does atomicAdd to
//   a 4-element global accumulator: [sum_mv_x, sum_mv_y, sum_mv_z, sum_m].
//
// Kernel 2: com_correct — subtracts COM velocity from all atoms.
//
// The accumulator must be zeroed before com_reduce (host-side memset or
// a trivial kernel). com_correct reads from the same accumulator.

// Accumulator layout: float[4] = {sum_mv_x, sum_mv_y, sum_mv_z, total_mass}

extern "C" __global__ void com_reduce(
    const float* __restrict__ velocities,  // [n_atoms * 3]
    const float* __restrict__ masses,      // [n_atoms]
    int n_atoms,
    float* __restrict__ accumulator         // [4] — atomicAdd target
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one atom's contribution
    float mv_x = 0.0f, mv_y = 0.0f, mv_z = 0.0f, m = 0.0f;
    if (tid < n_atoms) {
        m = masses[tid];
        mv_x = m * velocities[tid * 3];
        mv_y = m * velocities[tid * 3 + 1];
        mv_z = m * velocities[tid * 3 + 2];
    }

    // Warp-level reduction using shuffle
    unsigned int mask = __ballot_sync(0xFFFFFFFF, tid < n_atoms);
    for (int offset = 16; offset > 0; offset >>= 1) {
        mv_x += __shfl_down_sync(mask, mv_x, offset);
        mv_y += __shfl_down_sync(mask, mv_y, offset);
        mv_z += __shfl_down_sync(mask, mv_z, offset);
        m    += __shfl_down_sync(mask, m,    offset);
    }

    // Lane 0 of each warp writes to global accumulator via atomicAdd
    int lane = threadIdx.x & 31;
    if (lane == 0 && tid < n_atoms) {
        atomicAdd(&accumulator[0], mv_x);
        atomicAdd(&accumulator[1], mv_y);
        atomicAdd(&accumulator[2], mv_z);
        atomicAdd(&accumulator[3], m);
    }
}

extern "C" __global__ void com_correct(
    float* __restrict__ velocities,        // [n_atoms * 3] — modified in-place
    int n_atoms,
    const float* __restrict__ accumulator   // [4] — {sum_mv_x, sum_mv_y, sum_mv_z, total_mass}
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_atoms) return;

    float total_mass = accumulator[3];
    if (total_mass <= 0.0f) return;

    float inv_m = 1.0f / total_mass;
    float com_vx = accumulator[0] * inv_m;
    float com_vy = accumulator[1] * inv_m;
    float com_vz = accumulator[2] * inv_m;

    velocities[tid * 3]     -= com_vx;
    velocities[tid * 3 + 1] -= com_vy;
    velocities[tid * 3 + 2] -= com_vz;
}

// ─────────────────────────────────────────────────────────────────────────────
// com_accumulator_clear — zero the 4-element accumulator before reduction
// ─────────────────────────────────────────────────────────────────────────────
//
// Launch: <<<1, 1>>>

extern "C" __global__ void com_accumulator_clear(
    float* __restrict__ accumulator  // [4]
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        accumulator[0] = 0.0f;
        accumulator[1] = 0.0f;
        accumulator[2] = 0.0f;
        accumulator[3] = 0.0f;
    }
}
