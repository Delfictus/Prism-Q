// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / Rectification Phase 1 — Hard-Trap Invariant Enforcement
// ═══════════════════════════════════════════════════════════════════════
//
// Per the PRISM-4D Pipeline Rectification mandate §2 (operator
// directive 2026-04-29). Replaces soft-assertion error-code returns
// with HARDWARE-LEVEL traps via the PTX `trap` instruction. When an
// invariant violation is detected, the violating warp is terminated
// and the driver delivers a system-level interrupt that surfaces in
// the host as a non-recoverable CUDA error
// (`cudaErrorIllegalInstruction` / `cudaErrorLaunchFailure`).
//
// Why hard traps:
//
//   - On `printf`/error-code returns, downstream kernels can still
//     consume corrupted state before the host checks the error code.
//   - On `__trap()` / `asm volatile("trap;")`, the device terminates
//     IMMEDIATELY. No subsequent kernel on the same context can run
//     without a context reset. Corruption cannot propagate.
//   - The trap is a HARDWARE event on Blackwell (sm_120). Cannot be
//     suppressed or papered over by host code. The §9 ESCALATION
//     protocol is the ONLY recovery path.
//
// Compilation: nvcc -arch=sm_120 (PTX trap is supported on every
// CUDA architecture; sm_120 emits it natively without ucode wrap).
//
// ═══════════════════════════════════════════════════════════════════════

#ifndef PRISM_NHS_GPU_INVARIANT_CUH
#define PRISM_NHS_GPU_INVARIANT_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace prism_nhs { namespace gpu_invariant {

// ─────────────────────────────────────────────────────────────────────
// gpu_hard_assert — PTX trap if condition is false.
//
// Inline / forceinline so the trap is encoded into the parent
// kernel's PTX, not behind a function call. The compiler is free to
// optimize away the trap when `condition` is a compile-time-known
// true; it cannot optimize away a trap when `condition` is data-
// dependent (the volatile qualifier blocks instruction reordering
// across the trap).
//
// Behavior on violation:
//   - Warp executing the trap is terminated.
//   - Driver marks the launch as failed; subsequent
//     cudaStreamSynchronize / cudaDeviceSynchronize returns
//     cudaErrorIllegalInstruction (or cudaErrorLaunchFailure on
//     some driver versions).
//   - All other warps in the kernel are also killed (warp-group
//     teardown).
//   - The CUDA context is POISONED: no further work can be
//     scheduled until cudaDeviceReset() or context destruction.
// ─────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void gpu_hard_assert(bool condition) {
    if (!condition) {
        // PTX `trap` instruction. Identical idiom across sm_60+.
        asm volatile("trap;");
    }
}

// ─────────────────────────────────────────────────────────────────────
// gpu_hard_assert_eq_u64 — convenience overload for integer
// equality.
//
// Useful for the M1 Conservation-of-Mass invariant where two u64
// counters must satisfy a sum-equality condition.
// ─────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void gpu_hard_assert_eq_u64(
    unsigned long long lhs,
    unsigned long long rhs
) {
    gpu_hard_assert(lhs == rhs);
}

// ─────────────────────────────────────────────────────────────────────
// extern "C" host orchestration entry points.
//
// Definitions live in `gpu_invariant.cu`. Every function takes a
// `cudaStream_t` and returns `cudaError_t` via the `int32_t` ABI.
// ─────────────────────────────────────────────────────────────────────

extern "C" {

/// Sentinel `0xA55E47` (assert). Pinned by the Rust-side
/// `link_probe_returns_sentinel` test.
uint32_t prism_gpu_invariant_link_probe(void);

/// Launch the M1 Conservation-of-Mass audit kernel.
///
/// On stream `stream` reads
/// `*d_total_attributed + *d_background_count` and asserts equality
/// with `expected_total_input_spikes` via `gpu_hard_assert`. If the
/// assertion fails the device traps; the caller observes
/// `cudaStreamSynchronize` returning a non-success error.
///
/// **Caller contract**: after this returns `cudaSuccess`, the
/// caller MUST `cudaStreamSynchronize` (or otherwise drive the
/// stream to completion) BEFORE assuming the audit passed —
/// `cudaSuccess` here only confirms the kernel was queued.
cudaError_t prism_audit_mass_conservation(
    const uint64_t* d_total_attributed,
    const uint64_t* d_background_count,
    uint64_t expected_total_input_spikes,
    cudaStream_t stream
);

}  // extern "C"

}}  // namespace prism_nhs::gpu_invariant

#endif  // PRISM_NHS_GPU_INVARIANT_CUH
