// PRISM-4D Option A — DSTW variant-perturbation propagation kernel
// =================================================================
//
// Operator-authorised 2026-05-20 (PHASE 2: THE KERNEL GATE).
//
// Per-pair propagation of a rigid-backbone Δq/ΔV variant perturbation
// through the WT thermodynamic channel matrices K_active, K_lock,
// K_ensemble.  This is the GPU counterpart of the CPU reference
// projection in `crates/prism-nhs/src/dstw_dispatch/projection.rs`.
// The CPU reference used per-residue scalar weights; the GPU path
// upgrades to a real n_residues × n_residues per-pair matrix per
// channel, capturing how a perturbation at residue i propagates to
// the global thermodynamic signal via every other residue j.
//
// For a variant b with mutation at residue i_b and scalar perturbation
// p_b = α_q · Δq_b + α_v · ΔV_b:
//
//     ΔP_active(b)   = p_b · Σ_j K_active[i_b, j]
//     ΔP_lock(b)     = p_b · Σ_j K_lock[i_b, j]
//     ΔP_ensemble(b) = p_b · Σ_j K_ensemble[i_b, j]
//
// Each (variant, channel) pair becomes one CUDA block; the threads in
// that block cooperatively reduce the K row.  Inter-warp reduction uses
// `__shfl_down_sync` + a small shared-memory staging buffer so the kernel
// can scale from 4 warps (128 threads) up to 32 warps (1024 threads) on
// Blackwell (sm_120) without modification.
//
// Optimization targets (RTX 5080 / 5090, sm_120):
//   * Coalesced reads on K[i, j] — j is the inner (contiguous) dimension.
//   * Warp shuffles for intra-warp reduction (no shared bank conflicts).
//   * Shared memory only used for the cross-warp stage (32 floats per
//     block, negligible relative to the 128 KB/SM budget).
//   * No use_fast_math — engine policy is IEEE-correct division /
//     fma to avoid PTX codegen drift across builds.
//   * Pragma-unrolled inner loop (factor 4) so the issue rate keeps
//     pace with the L2 → registers bandwidth on Blackwell.
//
// Build path: registered in `crates/prism-gpu/build.rs` via
// `compile_kernel(..., "src/kernels/dstw_propagation.cu", ...)`.  The
// resulting `target/ptx/dstw_propagation.ptx` is loaded by the Rust FFI
// module at `crates/prism-nhs/src/dstw_dispatch/cuda.rs`.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

extern "C" {

// ---------------------------------------------------------------------------
// Constants exposed to the Rust FFI so test code can sweep matrix shapes
// without duplicating magic numbers.
// ---------------------------------------------------------------------------

// Number of thermodynamic channels emitted per variant (active/lock/ensemble).
__device__ __constant__ const int DSTW_N_CHANNELS = 3;

// Maximum supported residue count in a single dispatch.  Hard ceiling
// because the cross-warp reduction stage uses a 32-slot shared buffer
// (one slot per warp).  At 1024 threads/block we cover 1024 residues
// per block with a single pass; for larger receptors, repeat the kernel.
__device__ __constant__ const int DSTW_MAX_RESIDUES_PER_BLOCK = 1024;

// ---------------------------------------------------------------------------
// dstw_propagation_kernel
// ---------------------------------------------------------------------------
//
// Grid layout:
//   gridDim.x = n_variants (one block per variant)
//   gridDim.y = 3          (one block per channel)
//
// Block layout:
//   blockDim.x = power-of-two thread count, recommended 128-256
//   blockDim.y = 1
//
// Inputs:
//   K              -- [n_channels=3, n_residues, n_residues] float, row-major,
//                     channel-major outermost
//   residue_id     -- [n_variants] int, 0-indexed mutation site per variant
//   perturbation   -- [n_variants] float, α_q·Δq + α_v·ΔV per variant
//   n_variants     -- batch size
//   n_residues     -- receptor length (must be ≤ DSTW_MAX_RESIDUES_PER_BLOCK
//                     when blockDim.x ≥ n_residues; otherwise the inner loop
//                     strides through the row)
//
// Outputs:
//   delta_p        -- [n_channels=3, n_variants] float, row-major
//                     (channel-major, variant index inner)
//
// Behaviour on residue_id[b] out of range:
//   delta_p[c, b] = 0.0 for every channel c (DSTW will mark the variant
//   non-converged separately based on its own residue-range check; this
//   kernel never reads outside K and never traps).

__global__ void dstw_propagation_kernel(
    const float* __restrict__ K,
    const int*   __restrict__ residue_id,
    const float* __restrict__ perturbation,
    int n_variants,
    int n_residues,
    float* __restrict__ delta_p
) {
    int variant_idx = blockIdx.x;
    int channel = blockIdx.y;
    if (variant_idx >= n_variants || channel >= DSTW_N_CHANNELS) return;

    int i = residue_id[variant_idx];

    // Out-of-range residue: emit zero for this (variant, channel) pair.
    if (i < 0 || i >= n_residues) {
        if (threadIdx.x == 0) {
            delta_p[(size_t)channel * n_variants + variant_idx] = 0.0f;
        }
        return;
    }

    float pert = perturbation[variant_idx];
    size_t channel_stride = (size_t)n_residues * n_residues;
    const float* row_ptr = K + (size_t)channel * channel_stride
                            + (size_t)i * n_residues;

    // Strided cooperative sum of the row.  Inner loop unrolled 4x so the
    // memory pipeline on Blackwell stays saturated.
    float partial = 0.0f;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    #pragma unroll 4
    for (int j = tid; j < n_residues; j += stride) {
        partial += row_ptr[j];
    }

    // Intra-warp reduction via shuffles.  No shared-memory traffic.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    // Cross-warp reduction.  One lane-0 thread per warp writes to a
    // 32-slot shared scratch; warp 0 then reduces those slots.
    __shared__ float warp_sums[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = partial;
    }
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = blockDim.x >> 5;
        partial = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        }
        if (lane == 0) {
            delta_p[(size_t)channel * n_variants + variant_idx] = pert * partial;
        }
    }
}

// ---------------------------------------------------------------------------
// dstw_perturbation_assemble_kernel
// ---------------------------------------------------------------------------
//
// Helper kernel: assemble per-variant perturbation scalars on-device so
// the Rust host doesn't have to ship (α_q, α_v, Δq[b], ΔV[b]) → p_b
// across PCIe for every batch.  Trivial parallelism (one thread per
// variant) but pipelines neatly with the propagation kernel.
//
// Inputs:
//   delta_q       -- [n_variants] f32
//   delta_v       -- [n_variants] f32
//   alpha_q       -- scalar
//   alpha_v       -- scalar
//   n_variants
// Output:
//   perturbation  -- [n_variants] f32 = alpha_q*delta_q + alpha_v*delta_v

__global__ void dstw_perturbation_assemble_kernel(
    const float* __restrict__ delta_q,
    const float* __restrict__ delta_v,
    float alpha_q,
    float alpha_v,
    int n_variants,
    float* __restrict__ perturbation
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_variants) return;
    perturbation[b] = alpha_q * delta_q[b] + alpha_v * delta_v[b];
}

// ---------------------------------------------------------------------------
// dstw_epistemic_sigma_kernel
// ---------------------------------------------------------------------------
//
// Per-variant epistemic uncertainty assembly on-device.  Mirrors the
// CPU path in projection.rs:
//
//   sigma_c[b] = sqrt( pert[b]^2 · Var(K_c)[i_b]
//                    + K_row_sum_c[i_b]^2 · pert_var
//                    + model_residual_var_c )
//
// where Var(K_c)[i] is the engine's replicate-spread variance summed
// across the row, K_row_sum_c[i] = ΣⱼK_c[i,j], and pert_var =
// (α_q·σ_q)² + (α_v·σ_v)².
//
// The kernel is launched WITH `delta_p` already written by
// `dstw_propagation_kernel` so the host can read `K_row_sum_c[i_b]` as
// `delta_p[c, b] / pert[b]` when needed; here we pass them explicitly
// for numerical robustness (pert can be very small).

__global__ void dstw_epistemic_sigma_kernel(
    const float* __restrict__ row_sum_K,    // [n_channels, n_variants]
    const float* __restrict__ row_var_K,    // [n_channels, n_variants]
    const float* __restrict__ perturbation, // [n_variants]
    float pert_var,
    float model_residual_var_active,
    float model_residual_var_lock,
    float model_residual_var_ensemble,
    float nonconverged_penalty,             // 1.0 for converged, >1 for failed
    const uint8_t* __restrict__ converged_flag, // [n_variants] (0/1)
    int n_variants,
    float* __restrict__ sigma               // [n_channels, n_variants]
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    if (b >= n_variants || channel >= DSTW_N_CHANNELS) return;

    int idx = (size_t)channel * n_variants + b;
    float p = perturbation[b];
    float K_row = row_sum_K[idx];
    float K_var = row_var_K[idx];

    float residual = 0.0f;
    if (channel == 0)      residual = model_residual_var_active;
    else if (channel == 1) residual = model_residual_var_lock;
    else                   residual = model_residual_var_ensemble;

    float var = p * p * K_var + K_row * K_row * pert_var + residual;
    if (var < 0.0f) var = 0.0f;
    float s = sqrtf(var);

    if (converged_flag[b] == 0) {
        s *= nonconverged_penalty;
    }
    sigma[idx] = s;
}

} // extern "C"
