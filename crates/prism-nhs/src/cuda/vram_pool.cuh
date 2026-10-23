// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / F2 — Stream-Ordered Memory Pool + VRAM Audit shared header
// ═══════════════════════════════════════════════════════════════════════
//
// Per Blackwell Convergence mandate §3 (operator directive 2026-04-29).
// Replaces static "worst-case" pre-allocation with cudaMemPool-backed
// stream-ordered allocation and a global VRAM telemetry struct that
// the F1 Adjudicator's SWITCH node consumes as the Case-2 Violation
// trigger.
//
// **Layout contract (FFI-stable):**
//
//   VramAudit (24 bytes, 8-byte aligned):
//     u64 current_allocated_bytes  — offset 0
//     u64 peak_high_water_mark     — offset 8
//     u32 pool_exhaustion_flag     — offset 16  (0 = OK, 1 = ABORT)
//     [4 bytes trailing padding so arrays align]
//
//   The Rust-side mirror in `crates/prism-nhs/src/vram_pool.rs` has
//   matching `#[repr(C, align(8))]` and a `mem::size_of` pin test.
//
// **Concurrency model:**
//
//   Every telemetry update kernel runs as a single-thread launch
//   (`<<<1, 1, 0, stream>>>`) — the work is a constant-time atomic
//   chain, not a parallel reduction. Single-threaded launches keep
//   the semantics race-free under concurrent invocations from
//   different streams: every atomic op on the audit struct is a
//   single-instruction memory transaction, so cross-stream
//   concurrent updates compose correctly.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -c
//
// ═══════════════════════════════════════════════════════════════════════

#ifndef PRISM_NHS_VRAM_POOL_CUH
#define PRISM_NHS_VRAM_POOL_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace prism_nhs { namespace vram_pool {

// ─────────────────────────────────────────────────────────────────────
// VramAudit — 24-byte FFI-stable POD that lives in device memory.
//
// Three slots. Byte layout pinned by static_assert below.
// ─────────────────────────────────────────────────────────────────────
struct VramAudit {
    unsigned long long current_allocated_bytes;  // u64 at offset 0
    unsigned long long peak_high_water_mark;     // u64 at offset 8
    unsigned int       pool_exhaustion_flag;     // u32 at offset 16
};
static_assert(sizeof(VramAudit) == 24, "VramAudit FFI layout drift");
static_assert(alignof(VramAudit) == 8, "VramAudit FFI alignment drift");

// ─────────────────────────────────────────────────────────────────────
// Sentinel values for `pool_exhaustion_flag`.
// ─────────────────────────────────────────────────────────────────────
constexpr unsigned int VRAM_AUDIT_OK = 0u;
constexpr unsigned int VRAM_AUDIT_VIOLATION = 1u;

// ─────────────────────────────────────────────────────────────────────
// extern "C" host orchestration entry points.
//
// Definitions live in `vram_pool.cu`. Each function returns
// `cudaError_t` via the `int32_t` ABI (cudarc-side `CudaError = i32`).
//
// **Pool handle**: `cudaMemPool_t` is `void*` underneath. The Rust
// FFI carries it as `usize` and never dereferences it; only the
// `.cu` host code calls the runtime API. This keeps cudarc's safe
// surface clean and avoids re-exposing the runtime API in Rust.
//
// **Stream**: `cudaStream_t` is `void*`. Passed as `usize`.
//
// **VramAudit pointer**: device pointer to the 24-byte struct, also
// passed as `usize`. The Rust side allocates the struct via cudarc
// (`stream.alloc_zeros::<u8>(24)`) and threads its `device_ptr` here.
// ─────────────────────────────────────────────────────────────────────

extern "C" {

/// Sentinel `0xF2A4` (F2 audit). Pinned by the Rust-side
/// `link_probe_returns_sentinel` test.
uint32_t prism_vram_pool_link_probe(void);

/// Create a stream-ordered memory pool on `device_id`.
///
/// Configures the pool with:
///   - `cudaMemPoolAttrReleaseThreshold` = UINT64_MAX. The driver
///     never returns pool memory to the OS while the pool exists.
///     Critical for captured-graph execution: a release-threshold-
///     induced free during a graph replay would corrupt the graph's
///     mempool dependencies.
///   - `cudaMemPoolAttrReuseAllowOpportunistic` = 0. Reuse only
///     after stream-order completes, not opportunistically.
///   - `cudaMemPoolAttrReuseAllowInternalDependencies` = 0. Reuse
///     paths must respect the explicit stream graph, not heuristic
///     internal CUDA dependencies (defensive: avoids mempool
///     scheduling decisions that the captured graph cannot model).
///
/// Returns the pool handle through `*out_pool` as a `void*` cast to
/// `usize` on the Rust side.
cudaError_t prism_vram_pool_create(
    int32_t device_id,
    void** out_pool
);

/// Destroy a pool created by `prism_vram_pool_create`. After
/// destruction the handle is invalid; passing it to any other API
/// yields undefined behavior. Safe to call with a null handle (no-op).
cudaError_t prism_vram_pool_destroy(void* pool);

/// Allocate `size` bytes from `pool` on `stream`. Returns the device
/// pointer through `*out_ptr` (or null on failure).
cudaError_t prism_vram_pool_alloc_async(
    void*        pool,
    uint64_t     size,
    cudaStream_t stream,
    void**       out_ptr
);

/// Read-only attributes of this pool. Values are sampled sequentially, without
/// synchronizing streams or resetting high-water marks. These counters exclude
/// allocations owned by other pools, cudaMalloc, graphs, modules and contexts.
struct VramPoolMemorySnapshot {
    uint64_t used_current_bytes;
    uint64_t used_high_bytes;
    uint64_t reserved_current_bytes;
    uint64_t reserved_high_bytes;
};
static_assert(sizeof(VramPoolMemorySnapshot) == 32, "pool snapshot ABI drift");
static_assert(alignof(VramPoolMemorySnapshot) == 8, "pool snapshot alignment drift");
cudaError_t prism_vram_pool_memory_snapshot(
    void* pool, VramPoolMemorySnapshot* out_snapshot
);

/// Free `ptr` back to its pool on `stream`. Stream-ordered: the free
/// completes only after every operation already enqueued on `stream`
/// has retired (so kernels in flight that touch the buffer finish
/// before reuse).
cudaError_t prism_vram_pool_free_async(
    void*        ptr,
    cudaStream_t stream
);

/// Zero every field of `*d_audit`. Single-thread launch on `stream`.
cudaError_t prism_vram_audit_init(
    VramAudit*   d_audit,
    cudaStream_t stream
);

/// Telemetry update for one allocation. Single-thread launch on
/// `stream` that:
///   1. atomicAdd `alloc_size` onto `current_allocated_bytes`.
///   2. atomicMax `current_allocated_bytes` onto `peak_high_water_mark`.
///   3. If `current_allocated_bytes > budget`, atomicCAS the
///      `pool_exhaustion_flag` from 0 → 1 (idempotent on
///      already-set flag).
///
/// **Caller contract**: invoke this kernel IMMEDIATELY after every
/// `prism_vram_pool_alloc_async` (or its captured-graph node) so the
/// telemetry stays consistent with the actual allocation state.
cudaError_t prism_vram_audit_record_alloc(
    VramAudit*   d_audit,
    uint64_t     alloc_size,
    uint64_t     budget,
    cudaStream_t stream
);

/// Telemetry update for one free. Single-thread launch on `stream`
/// that subtracts `free_size` from `current_allocated_bytes` via the
/// two's-complement-additive trick (CUDA has no `atomicSub` for
/// u64). `peak_high_water_mark` is not modified.
cudaError_t prism_vram_audit_record_free(
    VramAudit*   d_audit,
    uint64_t     free_size,
    cudaStream_t stream
);

}  // extern "C"

}}  // namespace prism_nhs::vram_pool

#endif  // PRISM_NHS_VRAM_POOL_CUH
