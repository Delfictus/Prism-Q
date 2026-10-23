// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / WHILE Bounded Drain Bridge — Predicate Kernel FFI
// ═══════════════════════════════════════════════════════════════════════
//
// Begun 2026-05-04 — sub-ticket WHILE-POST-T7-DRAIN-001 (kernel portion).
// See `.prism_orchestration/WHILE_BOUNDED_MICROLOOP_EXECUTION_TICKET.md`.
//
// Background
// ----------
// CUDA 13.x exposes `cudaGraphCondTypeWhile` (driver_types.h:3560) — a
// graph conditional node type whose body subgraph re-executes while a
// device-mutable conditional handle is set non-zero (size==1, see
// :3572). The handle is mutated from inside the body via
// `cudaGraphSetConditional`. PRISM uses this for the post-T7 deferred-
// drain micro-loop: after the T7 sync barrier and before the monolithic
// child-graph splice, drain any deferred-error-state queue entries
// without reintroducing a host round-trip in the hot path.
//
// Predicate Bridge Pattern (mirrors gearbox.cu:436-477)
// -----------------------------------------------------
// A trivial single-thread `__global__` reads the device-resident pending
// counter, counts the body that just completed into the iteration counter,
// compares that completed count against a kernel-level `max_iterations` cap,
// and writes the resulting boolean (0 or 1) to the conditional handle via
// `cudaGraphSetConditional`. Counting occurs before continuation so the
// default-admitted first body is included in the exact device-side cap. This
// is the canonical "device-mutable predicate driving a WHILE conditional"
// pattern — parent-graph-resident, host_mutation=false, the
// max_iterations bound is enforced device-side so the host never has
// to interrupt the captured graph to break the loop.
//
// Forge-time topology
// -------------------
// The eventual wire-in (DEFERRED to a follow-up commit after R4 lands
// the F1 pipeline wiring) will:
//   1. Allocate `d_drain_pending: u32*` in the F2 VRAM pool, populated
//      by a small device-side write at the same site that increments
//      the host-side `Tier8DeferredDrainContext.drain_count`
//      (`fused_engine.rs:87`, mirror site).
//   2. Allocate `d_iter_count: u32*` initialized to 0 (memset before
//      WHILE entry; reset by the body subgraph each launch).
//   3. Create a WHILE conditional handle.
//   4. Launch this predicate kernel as the WHILE node's predicate.
//   5. Inside the body subgraph, perform one drain step + decrement
//      `d_drain_pending`.
//
// All five steps land in R4/R5/follow-up commits. THIS commit only
// stages the kernel + host shim + build entry so R5's WHILE FFI
// scaffold has a concrete predicate to reference.

#ifndef PRISM_WHILE_DRAIN_BRIDGE_CUH
#define PRISM_WHILE_DRAIN_BRIDGE_CUH

#include <cuda_runtime_api.h>
#include <cstdint>

// CUDA graph conditional handles are present in newer CUDA runtimes.
// The PrismOS laptop acceptance build may intentionally target an older
// toolkit to emit driver-compatible PTX for the installed GPU. Preserve the
// FFI ABI in that build; the implementation returns cudaErrorNotSupported
// when the runtime cannot execute WHILE conditionals.
#if !defined(CUDART_VERSION) || CUDART_VERSION < 12040
typedef unsigned long long cudaGraphConditionalHandle;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ─────────────────────────────────────────────────────────────────────
// prism_while_drain_predicate_kernel — device-mutable WHILE predicate
// ─────────────────────────────────────────────────────────────────────
//
// Single-thread `__global__` (`<<<1, 1, 0, stream>>>`) that:
//
//   pending   = *d_drain_pending               // queue depth (u32)
//   completed = atomicAdd(d_iter_count, 1) + 1 // body just completed
//   cont      = (pending > 0) && (completed < max_iterations) ? 1 : 0
//   cudaGraphSetConditional(handle, cont)
//
// The parent WHILE handle enters its first body from its default launch value,
// so the predicate runs after that body. Counting first makes `max_iterations`
// an exact physical-body cap and leaves `d_iter_count == max_iterations` at a
// max-bound exit.
//
// Properties (per ticket §Constraints / REJECT criteria):
//   • Parent-graph-resident — runs once per WHILE iteration as the
//     predicate, before the body subgraph body re-execution.
//   • host_mutation = false — host never writes `d_iter_count` or
//     `d_drain_pending` in the captured hot path; both are device-
//     resident and mutated by either this kernel (iter counter) or
//     the body subgraph drain step (pending counter).
//   • `max_iterations` is the hard cap enforced kernel-side. The host
//     pins this at forge time; it does NOT participate in the
//     captured graph and therefore cannot drift between launches.
//
// Args:
//   d_drain_pending   — device pointer to queue-depth u32 (read-only)
//   d_iter_count      — device pointer to iteration counter u32 (in/out)
//   handle            — WHILE conditional handle (size==1)
//   max_iterations    — host-pinned hard cap on iterations
extern "C" __global__ void prism_while_drain_predicate_kernel(
    const uint32_t*               d_drain_pending,
    uint32_t*                     d_iter_count,
    cudaGraphConditionalHandle    handle,
    uint32_t                      max_iterations
);

// ─────────────────────────────────────────────────────────────────────
// prism_while_drain_launch_predicate — host-side launcher
// ─────────────────────────────────────────────────────────────────────
//
// Launches `prism_while_drain_predicate_kernel` with `<<<1, 1, 0,
// stream>>>` and returns `cudaGetLastError()` cast to int. Mirrors the
// existing graph_node.cu / gearbox.cu return convention: 0 on
// success; nonzero `cudaError_t` otherwise.
//
// This shim is the symbol R5's WHILE FFI scaffold will reference when
// installing the predicate kernel as the WHILE conditional's predicate
// node. Callers (the eventual `captured_pipeline.rs` wire-in, deferred
// to a post-R4 follow-up commit) pass the WHILE handle obtained from
// `cudaGraphConditionalHandleCreate` and the device pointers from the
// F2 VRAM pool.
//
// Args:
//   stream            — capture stream the launch enters
//   d_drain_pending   — device pointer (read-only) to queue depth u32
//   d_iter_count      — device pointer to iteration counter u32 (in/out)
//   handle            — WHILE conditional handle
//   max_iterations    — host-pinned hard cap on iterations
//
// Returns: cudaError_t cast to int. 0 on success.
extern "C" int prism_while_drain_launch_predicate(
    cudaStream_t                  stream,
    const uint32_t*               d_drain_pending,
    uint32_t*                     d_iter_count,
    cudaGraphConditionalHandle    handle,
    uint32_t                      max_iterations
);

#ifdef __cplusplus
}
#endif

#endif  // PRISM_WHILE_DRAIN_BRIDGE_CUH
