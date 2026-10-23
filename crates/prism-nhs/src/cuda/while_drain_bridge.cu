// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / WHILE Bounded Drain Bridge — Predicate Kernel
// ═══════════════════════════════════════════════════════════════════════
//
// See `while_drain_bridge.cuh` for the full rationale and topology
// description. This TU implements the device-mutable predicate that
// drives the post-T7 deferred-drain WHILE micro-loop conditional.
//
// Compile target: sm_120 (RTX 5080 Blackwell). Gated on
// `CUDART_VERSION >= 12040` to match the ticket §1.2 floor where
// `cudaGraphSetConditional` and `cudaGraphCondTypeWhile` first ship.
// Production toolchain is CUDA 13.x so the live path is always taken;
// the disabled stub returns `cudaErrorNotSupported` for any older
// toolchain that might cross the build path.
//
// Mirrors the gearbox predicate-bridge pattern at
// `crates/prism-nhs/src/cuda/gearbox.cu:436-477` (single-thread
// `__global__`, `cudaGraphSetConditional` write, optional counter
// `atomicAdd`).

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstdint>

#include "while_drain_bridge.cuh"

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040

// ─────────────────────────────────────────────────────────────────────
// Predicate kernel — single-thread, device-mutable, parent-graph-resident
// ─────────────────────────────────────────────────────────────────────
//
//   pending   = *d_drain_pending
//   completed = atomicAdd(d_iter_count, 1) + 1
//   cont      = (pending > 0 && completed < max_iterations) ? 1 : 0
//   cudaGraphSetConditional(handle, cont)
//
// Single thread / single block — there is no parallelism to exploit
// across the predicate evaluation. Branchless on the condition: a
// single comparison + select; no warp divergence (only one thread
// participates in the launch).
//
// host_mutation = false: both `d_drain_pending` and `d_iter_count` are
// device-resident. The host never writes to them inside the captured
// hot path. `max_iterations` is host-pinned at forge time and travels
// in the kernel-launch parameters of the captured node — it does NOT
// participate in the graph mutation surface.
extern "C"
__global__ void prism_while_drain_predicate_kernel(
    const uint32_t*               d_drain_pending,
    uint32_t*                     d_iter_count,
    cudaGraphConditionalHandle    handle,
    uint32_t                      max_iterations)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (d_drain_pending == nullptr || d_iter_count == nullptr) return;

    const uint32_t pending = *d_drain_pending;

    // The parent WHILE handle defaults to true, so this predicate runs
    // *after* one complete body has already executed. Count that completed
    // body before deciding whether another body may enter. This keeps the
    // counter equal to the true completed-body count and makes a graph with
    // max_iterations=N execute exactly N bodies, including N=1.
    const uint32_t completed = atomicAdd(d_iter_count, 1u) + 1u;

    // Hard cap enforced kernel-side — the WHILE cannot iterate beyond
    // `max_iterations` even if the body subgraph fails to drain. This
    // is the watchdog that prevents runaway micro-loops; a separate
    // host-side audit emits `WhileExitReason::MaxIterations` when the
    // device counter equals the cap at exit.
    const uint32_t cont =
        (pending > 0u && completed < max_iterations) ? 1u : 0u;

    cudaGraphSetConditional(handle, cont);
}

extern "C"
int prism_while_drain_launch_predicate(
    cudaStream_t                  stream,
    const uint32_t*               d_drain_pending,
    uint32_t*                     d_iter_count,
    cudaGraphConditionalHandle    handle,
    uint32_t                      max_iterations)
{
    if (d_drain_pending == nullptr || d_iter_count == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    prism_while_drain_predicate_kernel<<<1, 1, 0, stream>>>(
        d_drain_pending,
        d_iter_count,
        handle,
        max_iterations);
    return static_cast<int>(cudaGetLastError());
}

// Routing only: preserve both existing voxel-pointer permutations and the
// exact native cold-reference cutoff inside one observation owner.
__global__ void prism_ladd_parent_selector_kernel(const uint32_t* step,
    uint32_t* parity, int32_t cold_end, cudaGraphConditionalHandle handle) {
    if (threadIdx.x || blockIdx.x) return;
    if (!step || !parity || *parity > 1u || *step > 0x7fffffffu) {
        asm volatile("trap;"); return;
    }
    const uint32_t branch = *parity + (static_cast<int32_t>(*step) < cold_end ? 0u : 2u);
    cudaGraphSetConditional(handle, branch);
    *parity ^= 1u;
}
extern "C" int prism_ladd_parent_launch_selector(cudaStream_t stream,
    const uint32_t* step, uint32_t* parity, int32_t cold_end, uint64_t handle) {
    if (!step || !parity) return static_cast<int>(cudaErrorInvalidValue);
    prism_ladd_parent_selector_kernel<<<1,1,0,stream>>>(step,parity,cold_end,
        static_cast<cudaGraphConditionalHandle>(handle));
    return static_cast<int>(cudaGetLastError());
}

#else  // CUDART_VERSION < 12040

// Pre-12.4 toolkits lack `cudaGraphSetConditional` /
// `cudaGraphCondTypeWhile`. Emit the host shim as a hard
// not-supported stub so the link surface stays stable; the device
// kernel symbol is omitted entirely (the stub is unreachable in
// production: nvcc 13.x is the only supported toolchain — see
// `build.rs:69` and ticket §1.2).

extern "C"
int prism_while_drain_launch_predicate(
    cudaStream_t               /*stream*/,
    const uint32_t*            /*d_drain_pending*/,
    uint32_t*                  /*d_iter_count*/,
    cudaGraphConditionalHandle /*handle*/,
    uint32_t                   /*max_iterations*/)
{
    return static_cast<int>(cudaErrorNotSupported);
}

extern "C" int prism_ladd_parent_launch_selector(cudaStream_t,
    const uint32_t*, uint32_t*, int32_t, uint64_t) {
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12040
