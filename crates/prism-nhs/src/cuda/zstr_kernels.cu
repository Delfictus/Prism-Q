// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / ZSTR — Zero-Stall Telemetry Ring kernels (impl)
//
// WP4 (ZSTR lossless persistence) revision.
//
// History: Amendment 3.4 (Path Z) moved the slot index from host-side
// `cuGraphExecKernelNodeSetParams_v2` node patching (broken under
// `cudaGraphAddChildGraphNode`) to a module-scope `__device__ uint32_t
// d_zstr_active_slot` updated from inside the captured graph body.
//
// WP4 finding: a module-scope symbol is ONE object per CUDA module, and all
// 8 telemetry streams share a single context + module.  Stream B's staging
// kernels could therefore consume a slot index written by stream A's
// slot-update kernel — silent cross-stream frame aliasing.  The symbol is
// DELETED.  The slot now lives in a PER-STREAM `uint32_t` cell inside that
// stream's own pinned ZstrRing control sector, and every ZSTR kernel takes
// an explicit pointer to it.
//
// The slot-update kernel additionally became the slot AUTHORITY: it applies
// bounded backpressure before claiming (so an unpersisted record can never
// be overwritten), and it stamps the frozen 4096-byte frame header, which
// no kernel previously wrote at all (the offline parser derived its file
// stride from `n_atoms` @ 20, read 0, and abandoned the file after the
// first header).
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════════

#include "zstr_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>  // Amendment 3.10 — kernel-level printf triage
#include <math_constants.h>

namespace prism_nhs { namespace zstr {

// ─── WP4 device helpers ─────────────────────────────────────────────────────

/// Load the per-stream active slot.  Volatile: the cell lives in pinned,
/// device-mapped host memory and is written by a kernel on a different
/// stream within the same graph body, so it must never be cached in a
/// register across the dependency edge.
__device__ __forceinline__ uint32_t zstr_load_active_slot(
    const uint32_t* __restrict__ d_active_slot
) {
    if (d_active_slot == nullptr) return PRISM_ZSTR_SLOT_NONE;
    return *reinterpret_cast<const volatile uint32_t*>(d_active_slot);
}

/// Block-wide broadcast of the active slot for the MULTI-BLOCK staging
/// kernels. The cell lives in device-mapped pinned host memory, so a
/// volatile load is a PCIe round trip that bypasses L1/L2 — having all
/// `n_atoms*3/4` threads issue one each would add thousands of uncached
/// host reads per epoch. Thread 0 pays the read once per block and
/// broadcasts through shared memory.
///
/// Every thread in the block MUST reach this call (the `__syncthreads()`
/// is unconditional), so it is invoked before any divergent early return.
__device__ __forceinline__ uint32_t zstr_broadcast_active_slot(
    const uint32_t* __restrict__ d_active_slot
) {
    __shared__ uint32_t s_slot;
    if (threadIdx.x == 0u) {
        s_slot = zstr_load_active_slot(d_active_slot);
    }
    __syncthreads();
    return s_slot;
}

__device__ __forceinline__ unsigned long long zstr_bp_load_u64(
    uint8_t* status, uint32_t off
) {
    return *reinterpret_cast<volatile unsigned long long*>(status + off);
}

__device__ __forceinline__ void zstr_bp_store_u64(
    uint8_t* status, uint32_t off, unsigned long long v
) {
    *reinterpret_cast<volatile unsigned long long*>(status + off) = v;
}

__device__ __forceinline__ uint32_t zstr_bp_load_u32(
    uint8_t* status, uint32_t off
) {
    return *reinterpret_cast<volatile uint32_t*>(status + off);
}

__device__ __forceinline__ void zstr_bp_store_u32(
    uint8_t* status, uint32_t off, uint32_t v
) {
    *reinterpret_cast<volatile uint32_t*>(status + off) = v;
}

// ─── Completion fence signal ────────────────────────────────────────────────

extern "C"
__global__ void zstr_signal_completion_kernel(
    uint8_t* __restrict__ base_fence,
    uint32_t              inter_slot_stride,
    const uint32_t* __restrict__ d_active_slot
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // WP4: an unclaimed epoch raises no fence.  Without this the host would
    // observe a READY slot whose payload was never staged (a phantom
    // record) or, worse, a re-fenced slot that still holds an unpersisted
    // prior record.
    const uint32_t slot = zstr_load_active_slot(d_active_slot);
    if (slot == PRISM_ZSTR_SLOT_NONE) return;

    // __threadfence_system() ensures all writes issued by ANY kernel in the
    // graph (positions, forces, adjudication_code) are globally visible —
    // including to the host ZSTR consumer — before the fence store executes.
    __threadfence_system();

    uint32_t* fence_ptr =
        reinterpret_cast<uint32_t*>(base_fence + slot * inter_slot_stride);

    atomicExch(fence_ptr, 1u);
}

// ─── Vectorized position staging ────────────────────────────────────────────

extern "C"
__global__ void zstr_pos_stage_f4_kernel(
    uint8_t* __restrict__       base_pinned,        // pinned slot-0 base
    uint32_t                    inter_slot_stride,   // bytes between slots
    uint32_t                    pos_offset_in_slot,  // header bytes inside slot
    const float4* __restrict__  src_vram,
    uint32_t                    n_floats,
    const uint32_t* __restrict__ d_active_slot
) {
    // Block-uniform: every thread reaches the broadcast before any return.
    const uint32_t slot = zstr_broadcast_active_slot(d_active_slot);
    if (slot == PRISM_ZSTR_SLOT_NONE) return;

    const uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n_f4 = (n_floats + 3u) >> 2u;
    if (tid >= n_f4) return;

    float4* dst = reinterpret_cast<float4*>(
        base_pinned + slot * inter_slot_stride + pos_offset_in_slot
    );

    // LDG.E.128 (cacheable global load via __ldg) → STG.E.128 non-temporal
    // store to pinned WC memory. Bypasses L2 on the write side so the PCIe
    // DMA engine sees the data immediately.
    float4 quad = __ldg(&src_vram[tid]);
    dst[tid] = quad;
}

// ─── T11 — Force-stage: vectorized DMA + warp-shuffle Σ‖F‖² atomic-add ─────

extern "C"
__global__ void zstr_force_stage_f4_kernel(
    uint8_t* __restrict__       base_pinned,
    uint32_t                    inter_slot_stride,
    uint32_t                    force_offset_in_slot,
    uint32_t                    force_norm_offset_in_slot,
    const float4* __restrict__  src_forces,
    uint32_t                    n_floats,
    const uint32_t* __restrict__ d_active_slot
) {
    // Block-uniform: reached by every thread, so the __shfl_down_sync
    // butterfly below still runs on full warps.
    const uint32_t slot = zstr_broadcast_active_slot(d_active_slot);
    if (slot == PRISM_ZSTR_SLOT_NONE) return;

    const uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t n_f4 = (n_floats + 3u) >> 2u;

    uint8_t*  slot_base = base_pinned + slot * inter_slot_stride;
    float4*   dst       = reinterpret_cast<float4*>(slot_base + force_offset_in_slot);
    float*    fn_ptr    = reinterpret_cast<float*>(slot_base + force_norm_offset_in_slot);

    // Per-thread DMA + sum-of-squares.  The tail bounds-check ensures
    // the last quad's components beyond n_floats do NOT contribute; we
    // still STG the quad verbatim (the trailing slots in the pinned
    // forces region are unused) — keeping the store fully vectorised.
    float local_sumsq = 0.0f;
    if (tid < n_f4) {
        const float4 q = __ldg(&src_forces[tid]);
        dst[tid] = q;

        const uint32_t base_idx = tid * 4u;
        if (base_idx + 0u < n_floats) local_sumsq += q.x * q.x;
        if (base_idx + 1u < n_floats) local_sumsq += q.y * q.y;
        if (base_idx + 2u < n_floats) local_sumsq += q.z * q.z;
        if (base_idx + 3u < n_floats) local_sumsq += q.w * q.w;
    }

    // Warp-level butterfly reduction (32-lane).  __shfl_down_sync is the
    // sm_120 SOTA primitive — single-instruction inter-lane communication
    // with no shared-memory traffic.  After the loop, lane 0 holds Σ over
    // its warp's 32 threads.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sumsq += __shfl_down_sync(0xffffffffu, local_sumsq, offset);
    }

    // Each warp's lane-0 atomicAdds its partial sum directly into the
    // pinned host slot's force_norm field — no intermediate scratch.
    // Pinned-host atomicAdd lands as a system-scope PCIe atomic on
    // sm_120; the host Reaper reads the post-sqrt finalised value.
    if (lane == 0u) {
        atomicAdd(fn_ptr, local_sumsq);
    }
}

// ─── T11 — Single-thread post-pass: in-place sqrtf of force_norm ──────────

extern "C"
__global__ void zstr_force_norm_sqrt_kernel(
    uint8_t* __restrict__       base_pinned,
    uint32_t                    inter_slot_stride,
    uint32_t                    force_norm_offset_in_slot,
    const uint32_t* __restrict__ d_active_slot,
    // P8 — POINTER-STABLE FORCE-NORM MIRROR (nullable).
    //
    // The authoritative force_norm lives in the ZSTR ring slot, and the slot
    // rolls `step % 5`. The G26 governor's force guard binds ONE device
    // pointer for the life of the captured graph, so it cannot read a
    // rolling address: a fixed binding would be correct one step in five.
    //
    // Without a stable address `PrismGearGuardObservation::force_measured`
    // stays 0, and gate 5 then pins the ceiling to the CURRENT gear whenever
    // `require_force_guard_for_upshift` is set (it defaults to 1). That makes
    // the force guard inert in BOTH directions — it cannot authorise an
    // upshift and it cannot see a force excursion to downshift on.
    //
    // This mirror is written in the same single-thread pass that already owns
    // the sqrt, so it costs one store and introduces no new synchronisation
    // and no new kernel.
    float* __restrict__         d_force_norm_mirror
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint32_t slot = zstr_load_active_slot(d_active_slot);
    if (slot == PRISM_ZSTR_SLOT_NONE) {
        // P8: no slot claimed this step means no measurement exists. The
        // mirror is left untouched rather than zeroed — a zero force norm is
        // exactly what a frozen system would report, and writing one here
        // would let "nothing was staged" read as "forces are quiescent".
        return;
    }

    float* fn_ptr = reinterpret_cast<float*>(
        base_pinned + slot * inter_slot_stride + force_norm_offset_in_slot
    );

    // sqrtf propagates NaN verbatim; negative values (impossible on a
    // sum-of-squares but defensive for clamp drift) produce NaN under
    // IEEE-754 — the host G29 Reaper traps non-finite reads.
    const float sumsq = *fn_ptr;
    const float norm  = sqrtf(sumsq);
    *fn_ptr = norm;

    // P8 — publish to the pointer-stable mirror the governor binds.
    if (d_force_norm_mirror != nullptr) {
        *d_force_norm_mirror = norm;
    }
}

// ─── M1.2.18.5 — Single-thread audit-field stage kernel ─────────────────────
//
// Reads V_t (f64) from `adj.potential_energy` at FFI offset 112 and W_ext
// (f64) by dereferencing the *mut f64 pointer at FFI offset 128.  Writes
// the two f64s into the active ZSTR slot at the operator-specified offsets:
//
//     slot[external_work_offset_in_slot]   ← W_ext     (offset 32, 8 B)
//     slot[potential_energy_offset_in_slot] ← V_t      (offset 40, 8 B)
//
// WP4: also stamps `adjudication_code` (slot offset 12) from `adj + 52`
// (pinned by adjudicator.cuh's static_assert, same offset gearbox.cu reads).
// This is the only ZSTR node captured downstream of the Adjudicator, so it
// is the sole owner of that header field; before WP4 nothing wrote it and
// the offline parser always reported code 0.
//
// Captured downstream of the SFA so V_t and W_ext are settled before
// the read.  Pinned-host writes land as system-scope PCIe stores on
// sm_120, visible to the Reaper as soon as completion_fence flips.
//
// `adj` may be null in legacy / test fixtures — kernel writes 0.0 to
// both fields in that case (and leaves adjudication_code at its stamped 0).
extern "C"
__global__ void zstr_stage_audit_kernel(
    uint8_t* __restrict__   base_pinned,
    uint32_t                inter_slot_stride,
    uint32_t                external_work_offset_in_slot,
    uint32_t                potential_energy_offset_in_slot,
    uint32_t                adjudication_code_offset_in_slot,
    const uint8_t* __restrict__ adj,
    const uint32_t* __restrict__ d_active_slot
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint32_t slot = zstr_load_active_slot(d_active_slot);
    if (slot == PRISM_ZSTR_SLOT_NONE) return;

    uint8_t* slot_base  = base_pinned + slot * inter_slot_stride;

    double   v_t      = 0.0;
    double   w_ext    = 0.0;
    uint32_t adj_code = 0u;
    if (adj != nullptr) {
        // FFI offset 112: potential_energy (f64 VALUE).
        v_t = *reinterpret_cast<const double*>(adj + 112);
        // FFI offset 128: d_external_work (*mut f64 POINTER).
        const double* w_ptr =
            *reinterpret_cast<const double* const*>(adj + 128);
        if (w_ptr != nullptr) {
            w_ext = *w_ptr;
        }
        // FFI offset 52: adjudication_code (u32) — 0 Prune / 1 Construct /
        // 2 Violation.
        adj_code = *reinterpret_cast<const uint32_t*>(adj + 52);
    }

    *reinterpret_cast<double*>(slot_base + external_work_offset_in_slot)
        = w_ext;
    *reinterpret_cast<double*>(slot_base + potential_energy_offset_in_slot)
        = v_t;
    *reinterpret_cast<uint32_t*>(slot_base + adjudication_code_offset_in_slot)
        = adj_code;
}

}} // namespace prism_nhs::zstr

// ─── C-ABI capture-window launchers ─────────────────────────────────────────

extern "C"
int zstr_launch_pos_stage(void*       base_pinned,
                           uint32_t    inter_slot_stride,
                           uint32_t    pos_offset_in_slot,
                           const void* src_vram,
                           uint32_t    n_atoms,
                           const void* d_active_slot,
                           void*       stream)
{
    const uint32_t n_floats = n_atoms * 3u;
    const uint32_t n_f4     = (n_floats + 3u) >> 2u;
    const uint32_t block    = 256u;
    const uint32_t grid     = (n_f4 + block - 1u) / block;

    prism_nhs::zstr::zstr_pos_stage_f4_kernel<<<grid, block, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint8_t*>(base_pinned),
        inter_slot_stride,
        pos_offset_in_slot,
        static_cast<const float4*>(src_vram),
        n_floats,
        static_cast<const uint32_t*>(d_active_slot)
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C"
int zstr_launch_fence_signal(void*    base_fence,
                              uint32_t inter_slot_stride,
                              const void* d_active_slot,
                              void*    stream)
{
    prism_nhs::zstr::zstr_signal_completion_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint8_t*>(base_fence),
        inter_slot_stride,
        static_cast<const uint32_t*>(d_active_slot)
    );
    return static_cast<int>(cudaGetLastError());
}

// ─── DEPRECATED host-side slot index update (per-stream pointer form) ───────
//
// WP4: the module-scope `d_zstr_active_slot` symbol is gone, so
// `cudaMemcpyToSymbolAsync` has no target.  The capability (host chooses an
// explicit slot, stream-ordered) is preserved here in per-stream pointer
// form.  A single-thread kernel is used rather than cudaMemcpyAsync because
// the source would otherwise be a pageable host stack local, which is not
// legal for a stream-ordered async copy.
//
// NOT used by the production path: only the in-graph slot-authority kernel
// also stamps the frame header, and a claim without a stamp reproduces the
// WP4 defect-1 unparseable-file failure.

namespace prism_nhs { namespace zstr {

extern "C" __global__
void prism_zstr_set_active_slot_kernel(uint32_t* d_active_slot, uint32_t slot)
{
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    if (d_active_slot == nullptr) return;
    *reinterpret_cast<volatile uint32_t*>(d_active_slot) = slot;
    __threadfence_system();
}

}} // namespace prism_nhs::zstr

extern "C"
int prism_zstr_set_active_slot_ptr(void* d_active_slot, uint32_t slot, void* stream)
{
    prism_nhs::zstr::prism_zstr_set_active_slot_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint32_t*>(d_active_slot),
        slot
    );
    return static_cast<int>(cudaGetLastError());
}

// ═══════════════════════════════════════════════════════════════════════════
// WP4 §A — ZSTR SLOT AUTHORITY (backpressure + claim + header stamp)
// ═══════════════════════════════════════════════════════════════════════════
//
// Single-thread `__global__` kernel recorded FIRST in every captured
// graph-body iteration.  Topology order inside the unrolled body:
//   §1  increment_time_kernel      -> *d_step_counter += 1
//   WP4 device_slot_update         -> backpressure + claim + header stamp
//   physics integration / SO(3) / Adjudicator / ASC nodes
//   ZSTR snapshot kernels          (read *d_active_slot)
//   ZSTR fence signal              (read *d_active_slot)
//   Ghost v2 emission              (reads *d_step_counter via §1)
//
// Pre-condition: the §1 device step counter (uint64_t* d_step_counter,
// allocated by nhs_rt_full at V2-BUILD seeded with v2_trigger_step) must
// have been incremented by `prism_increment_time_kernel` earlier in the
// captured body of the same iteration.
//
// Backpressure contract (Gate Rule 4 — every wait is bounded and fails
// loudly):
//   * The wait target is the claimed slot's `completion_fence`.  Non-zero
//     means the host consumer has NOT yet persisted the record currently
//     resident in that slot.
//   * The wait is bounded by `backpressure_spin_limit` iterations, each of
//     which is a PCIe read of pinned host memory plus a ~1 µs __nanosleep,
//     so the limit is approximately a microsecond budget.
//   * On expiry the kernel records PRISM_ZSTR_BP_ERR_SPIN_EXPIRED plus the
//     offending slot in the per-stream status block, bumps the
//     backpressure_events counter, writes PRISM_ZSTR_SLOT_NONE into
//     `*d_active_slot`, and returns.  Nothing downstream stages, nothing
//     fences, no record is overwritten and no epoch is silently lost — the
//     host receipt shows epochs > produced by exactly the event count.
//   * The error code is STICKY and short-circuits the wait on subsequent
//     epochs.  A drain path that has already blown a multi-second budget is
//     not going to recover within the same launch, and paying the full
//     budget on every one of the (unroll × chunks) remaining epochs would
//     convert a bounded data-loss event into an unbounded wall-clock hang
//     on a paid GPU lease.

namespace prism_nhs { namespace zstr {

extern "C" __global__
void prism_zstr_device_slot_update_kernel(
    const unsigned long long* __restrict__ d_step_counter,
    uint32_t              n_slots,
    uint8_t* __restrict__ base_pinned,
    uint32_t              inter_slot_stride,
    uint32_t* __restrict__ d_active_slot,
    uint32_t              n_atoms,
    float                 dt,
    uint32_t              gear_id,
    uint32_t              stream_id,
    uint32_t              schema_version,
    uint32_t              backpressure_spin_limit,
    uint8_t* __restrict__ d_backpressure_status)
{
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;

    // ZSTR not wired into this build/stream — nothing to own.
    if (d_active_slot == nullptr) return;

    volatile uint32_t* slot_cell =
        reinterpret_cast<volatile uint32_t*>(d_active_slot);

    uint8_t* status = d_backpressure_status;

    if (d_step_counter == nullptr || n_slots == 0u || base_pinned == nullptr) {
        *slot_cell = PRISM_ZSTR_SLOT_NONE;
        if (status != nullptr) {
            zstr_bp_store_u32(status, PRISM_ZSTR_BP_OFF_ERROR,
                              PRISM_ZSTR_BP_ERR_NULL_ARG);
        }
        __threadfence_system();
        return;
    }

    // Epoch accounting.  This kernel is single-thread and serialised against
    // itself by the captured-graph body dependency chain, and the status
    // block is per-stream, so a plain volatile RMW is race-free.
    if (status != nullptr) {
        zstr_bp_store_u64(status, PRISM_ZSTR_BP_OFF_EPOCHS,
                          zstr_bp_load_u64(status, PRISM_ZSTR_BP_OFF_EPOCHS) + 1ull);
    }

    const unsigned long long step = *d_step_counter;
    const uint32_t slot =
        static_cast<uint32_t>(step % static_cast<unsigned long long>(n_slots));

    // ── 1. Bounded backpressure ────────────────────────────────────────────
    volatile uint32_t* fence_cell = reinterpret_cast<volatile uint32_t*>(
        base_pinned
        + static_cast<size_t>(slot) * static_cast<size_t>(inter_slot_stride)
        + PRISM_ZSTR_HDR_OFF_FENCE
    );

    const bool sticky_failed =
        (status != nullptr)
        && (zstr_bp_load_u32(status, PRISM_ZSTR_BP_OFF_ERROR)
            != PRISM_ZSTR_BP_ERR_NONE);

    if (backpressure_spin_limit != 0u) {
        uint32_t spins = 0u;
        while (*fence_cell != 0u) {
            if (sticky_failed || spins >= backpressure_spin_limit) {
                // Refuse the claim. Never overwrite an unpersisted record.
                *slot_cell = PRISM_ZSTR_SLOT_NONE;
                if (status != nullptr) {
                    zstr_bp_store_u32(status, PRISM_ZSTR_BP_OFF_ERROR,
                                      PRISM_ZSTR_BP_ERR_SPIN_EXPIRED);
                    zstr_bp_store_u32(status, PRISM_ZSTR_BP_OFF_SLOT, slot);
                    zstr_bp_store_u64(
                        status, PRISM_ZSTR_BP_OFF_EVENTS,
                        zstr_bp_load_u64(status, PRISM_ZSTR_BP_OFF_EVENTS) + 1ull);
                }
                __threadfence_system();
                return;
            }
            ++spins;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
            __nanosleep(1000u);  // ~1 µs; keeps the PCIe poll rate sane
#else
            __threadfence_block();
#endif
        }
    }

    // ── 2. Claim ───────────────────────────────────────────────────────────
    uint8_t* slot_base =
        base_pinned + static_cast<size_t>(slot) * static_cast<size_t>(inter_slot_stride);

    // ── 3. Stamp the frozen header ─────────────────────────────────────────
    // frame_idx carries the AUTHORITATIVE device step so the host consumer
    // can order records by production time rather than by slot permutation,
    // and so the offline parser can recover the file stride from n_atoms.
    // completion_fence (16) / force_norm (28) / external_work (32) /
    // potential_energy (40) are deliberately untouched: they are owned by
    // the fence, force-stage and audit kernels.
    *reinterpret_cast<volatile unsigned long long*>(
        slot_base + PRISM_ZSTR_HDR_OFF_FRAME_IDX) = step;
    *reinterpret_cast<volatile float*>(
        slot_base + PRISM_ZSTR_HDR_OFF_DT) = dt;
    // adjudication_code is re-stamped downstream by zstr_stage_audit_kernel
    // (the only ZSTR node captured after the Adjudicator).  Zero here so a
    // slot whose audit node is absent never carries a stale prior code.
    *reinterpret_cast<volatile uint32_t*>(
        slot_base + PRISM_ZSTR_HDR_OFF_ADJ_CODE) = 0u;
    *reinterpret_cast<volatile uint32_t*>(
        slot_base + PRISM_ZSTR_HDR_OFF_N_ATOMS) = n_atoms;
    *reinterpret_cast<volatile uint32_t*>(
        slot_base + PRISM_ZSTR_HDR_OFF_GEAR_ID) = gear_id;
    *reinterpret_cast<volatile uint32_t*>(
        slot_base + PRISM_ZSTR_HDR_OFF_STREAM_ID) = stream_id;
    *reinterpret_cast<volatile uint32_t*>(
        slot_base + PRISM_ZSTR_HDR_OFF_SCHEMA_VER) = schema_version;

    __threadfence_system();

    *slot_cell = slot;

    if (status != nullptr) {
        zstr_bp_store_u64(status, PRISM_ZSTR_BP_OFF_PRODUCED,
                          zstr_bp_load_u64(status, PRISM_ZSTR_BP_OFF_PRODUCED) + 1ull);
    }

    __threadfence_system();
}

}} // namespace prism_nhs::zstr

extern "C"
int prism_zstr_device_slot_update_launch(const void* d_step_counter,
                                          uint32_t    n_slots,
                                          void*       base_pinned,
                                          uint32_t    inter_slot_stride,
                                          void*       d_active_slot,
                                          uint32_t    n_atoms,
                                          float       dt,
                                          uint32_t    gear_id,
                                          uint32_t    stream_id,
                                          uint32_t    schema_version,
                                          uint32_t    backpressure_spin_limit,
                                          void*       d_backpressure_error,
                                          void*       stream)
{
    prism_nhs::zstr::prism_zstr_device_slot_update_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const unsigned long long*>(d_step_counter),
        n_slots,
        static_cast<uint8_t*>(base_pinned),
        inter_slot_stride,
        static_cast<uint32_t*>(d_active_slot),
        n_atoms,
        dt,
        gear_id,
        stream_id,
        schema_version,
        backpressure_spin_limit,
        static_cast<uint8_t*>(d_backpressure_error)
    );
    return static_cast<int>(cudaGetLastError());
}

// ─── T11 — Force-stage launchers ────────────────────────────────────────────

extern "C"
int zstr_launch_force_stage(void*       base_pinned,
                             uint32_t    inter_slot_stride,
                             uint32_t    force_offset_in_slot,
                             uint32_t    force_norm_offset_in_slot,
                             const void* src_d_forces,
                             uint32_t    n_atoms,
                             const void* d_active_slot,
                             void*       stream)
{
    const uint32_t n_floats = n_atoms * 3u;
    const uint32_t n_f4     = (n_floats + 3u) >> 2u;
    const uint32_t block    = 256u;
    const uint32_t grid     = (n_f4 + block - 1u) / block;

    prism_nhs::zstr::zstr_force_stage_f4_kernel<<<grid, block, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint8_t*>(base_pinned),
        inter_slot_stride,
        force_offset_in_slot,
        force_norm_offset_in_slot,
        static_cast<const float4*>(src_d_forces),
        n_floats,
        static_cast<const uint32_t*>(d_active_slot)
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C"
int zstr_launch_force_norm_sqrt(void*    base_pinned,
                                 uint32_t inter_slot_stride,
                                 uint32_t force_norm_offset_in_slot,
                                 const void* d_active_slot,
                                 void*    stream,
                                 void*    d_force_norm_mirror)
{
    prism_nhs::zstr::zstr_force_norm_sqrt_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint8_t*>(base_pinned),
        inter_slot_stride,
        force_norm_offset_in_slot,
        static_cast<const uint32_t*>(d_active_slot),
        static_cast<float*>(d_force_norm_mirror)
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C"
int zstr_launch_stage_audit(void*       base_pinned,
                             uint32_t    inter_slot_stride,
                             uint32_t    external_work_offset_in_slot,
                             uint32_t    potential_energy_offset_in_slot,
                             uint32_t    adjudication_code_offset_in_slot,
                             const void* adj,
                             const void* d_active_slot,
                             void*       stream)
{
    prism_nhs::zstr::zstr_stage_audit_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint8_t*>(base_pinned),
        inter_slot_stride,
        external_work_offset_in_slot,
        potential_energy_offset_in_slot,
        adjudication_code_offset_in_slot,
        static_cast<const uint8_t*>(adj),
        static_cast<const uint32_t*>(d_active_slot)
    );
    return static_cast<int>(cudaGetLastError());
}
