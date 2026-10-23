// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / ZSTR — Zero-Stall Telemetry Ring kernel declarations
//
// Kernel roles:
//   1. prism_zstr_device_slot_update — WP4 slot authority.  Runs FIRST in
//      every captured graph-body iteration.  Applies bounded backpressure,
//      claims the slot into a PER-STREAM `uint32_t* d_active_slot`, and
//      stamps the frozen 4096-byte frame header (frame_idx / dt / n_atoms /
//      gear_id / adjudication_code / stream_id / schema_version).
//
//   2. zstr_signal_completion  — fires at end of captured CUDA graph DAG;
//      __threadfence_system() + pinned-fence write notifies the host
//      ZSTR consumer without any cudaStreamSynchronize.
//
//   3. zstr_pos_stage_f4 / zstr_force_stage_f4 / zstr_force_norm_sqrt /
//      zstr_stage_audit — payload + observable staging into the claimed
//      slot.
//
// WP4 (ZSTR lossless persistence) — cross-stream aliasing eradication.
// The module-scope `__device__ uint32_t d_zstr_active_slot` symbol has been
// DELETED.  All 8 telemetry streams share one CUDA context and one module,
// so a single module-scope symbol let stream B's kernels consume a slot
// index written by stream A.  Every ZSTR kernel now takes an explicit
// per-stream `d_active_slot` pointer (host-allocated in the stream's own
// ZstrRing control sector), which makes the slot binding structurally
// unshareable.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -c
// ═══════════════════════════════════════════════════════════════════════════

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ─── WP4 shared layout constants ────────────────────────────────────────────
//
// Sentinel written into `*d_active_slot` when NO slot has been claimed for
// the current epoch (backpressure expiry, or null/degenerate arguments).
// Every downstream ZSTR kernel treats it as "skip this epoch entirely" so a
// missed claim can never overwrite an unpersisted record.
#define PRISM_ZSTR_SLOT_NONE 0xFFFFFFFFu

// Frozen `ZstrFrameHeader` byte offsets (crates/prism-nhs/src/zstr.rs).
// The struct is `#[repr(C, align(4096))]` and pinned by a compile-time
// assert; these constants and the offline parser
// (bin/prism_ghost_evidence_scan.rs) are the two external consumers.
#define PRISM_ZSTR_HDR_OFF_FRAME_IDX  0u   // u64 — authoritative device step
#define PRISM_ZSTR_HDR_OFF_DT         8u   // f32 — timestep, picoseconds
#define PRISM_ZSTR_HDR_OFF_ADJ_CODE   12u  // u32 — adjudication_code
#define PRISM_ZSTR_HDR_OFF_FENCE      16u  // u32 — completion_fence
#define PRISM_ZSTR_HDR_OFF_N_ATOMS    20u  // u32 — atom count (file stride)
#define PRISM_ZSTR_HDR_OFF_GEAR_ID    24u  // u32 — G26 chronometric gear

// PRISM 2026-08-02 — gear_id sentinel for "not measured".
//
// The documented gear encoding is 0 = 0.5 fs, 1 = 2.0 fs, 2 = 4.0 fs (HMR),
// 3 = abort-trap. The V2 config currently supplies a HARDCODED gear_id of 0
// (nhs_rt_full.rs, "Wave A default"), and the device-side cruise->current_gear
// never leaves the GPU, so no measured gear reaches the host at all.
//
// Stamping that literal 0 into the frame header would assert "this frame was
// integrated at 0.5 fs" on every record, while the run is actually at 2.0 fs
// (or 4.0 fs under HMR) — a 4x-to-8x misstatement of temporal resolution,
// recorded as if it were measurement. Offline consumers cannot distinguish a
// fabricated 0 from a measured gear 0.
//
// Until the gearbox publishes a real gear to the host, stamp this sentinel so
// the absence of a measurement is explicit and unmistakable.
#define PRISM_ZSTR_GEAR_ID_UNMEASURED 0xFFFFFFFFu
#define PRISM_ZSTR_HDR_OFF_STREAM_ID  48u  // u32 — WP4: producing stream id
#define PRISM_ZSTR_HDR_OFF_SCHEMA_VER 52u  // u32 — WP4: header schema version

// Per-stream backpressure/conservation status block.  Lives in the ZstrRing
// control sector (pinned + DEVICEMAP, 4096-aligned) so the device writes it
// and the host consumer/receipt reads it with NO CUDA API call.
//
//   +0  u32 error_code       (PRISM_ZSTR_BP_ERR_*)
//   +4  u32 error_slot       (slot that expired; valid iff error_code != 0)
//   +8  u64 epochs           (slot-update kernel invocations)
//   +16 u64 produced         (successful claims — records staged + fenced)
//   +24 u64 backpressure_events (claims refused by the bounded wait)
//
// Written ONLY by the single-thread slot-update kernel, which is serialised
// against itself by the captured-graph body dependency chain, so plain
// volatile read-modify-write is race-free (no 64-bit system atomics needed).
#define PRISM_ZSTR_BP_OFF_ERROR    0u
#define PRISM_ZSTR_BP_OFF_SLOT     4u
#define PRISM_ZSTR_BP_OFF_EPOCHS   8u
#define PRISM_ZSTR_BP_OFF_PRODUCED 16u
#define PRISM_ZSTR_BP_OFF_EVENTS   24u

#define PRISM_ZSTR_BP_ERR_NONE         0u
#define PRISM_ZSTR_BP_ERR_SPIN_EXPIRED 1u
#define PRISM_ZSTR_BP_ERR_NULL_ARG     2u

namespace prism_nhs { namespace zstr {

/// WP4 §A — slot authority.  Single-thread <<<1,1>>>, recorded FIRST in
/// each captured graph-body iteration on the stream's md_stream.
///
/// Three jobs, in order:
///   1. Bounded backpressure — spin on the target slot's `completion_fence`
///      until the host consumer has cleared it, bounded by
///      `backpressure_spin_limit`.  On expiry: record the error in
///      `d_backpressure_status`, write PRISM_ZSTR_SLOT_NONE into
///      `d_active_slot`, and RETURN WITHOUT CLAIMING.  Never overwrites an
///      unpersisted record; never hangs.
///   2. Claim `(*d_step_counter) % n_slots` into the per-stream
///      `*d_active_slot`.
///   3. Stamp the frozen header for that slot (frame_idx = the device step,
///      dt, n_atoms, gear_id, adjudication_code=0, stream_id,
///      schema_version).  `completion_fence`, `force_norm`, `external_work`
///      and `potential_energy` are NOT touched — they belong to the fence,
///      force-stage and audit kernels respectively.
extern "C"
__global__ void prism_zstr_device_slot_update_kernel(
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
    uint8_t* __restrict__ d_backpressure_status
);

/// Written at the end of every captured-graph epoch.
/// Sets slot_fence to 1 after __threadfence_system() ensures all
/// preceding DMA writes (positions, forces) are visible to the host.
///
/// WP4: the slot comes from the per-stream `d_active_slot`; a
/// PRISM_ZSTR_SLOT_NONE value means the epoch never claimed a slot and the
/// fence is not raised (no phantom record).
extern "C"
__global__ void zstr_signal_completion_kernel(
    uint8_t* __restrict__ base_fence,
    uint32_t              inter_slot_stride,
    const uint32_t* __restrict__ d_active_slot
);

/// WP4 per-stream-slot variant: kernel reads `*d_active_slot` at execution
/// time to compute the slot-specific positions destination inside the pinned
/// ring.  base_pinned points at slot-0's header start; pos_offset_in_slot
/// adds the header-size offset to land on the positions payload.
extern "C"
__global__ void zstr_pos_stage_f4_kernel(
    uint8_t* __restrict__       base_pinned,
    uint32_t                    inter_slot_stride,
    uint32_t                    pos_offset_in_slot,
    const float4* __restrict__  src_vram,
    uint32_t                    n_floats,
    const uint32_t* __restrict__ d_active_slot
);

/// T11 — Action-Recovery force-stage kernel.
///
/// Vectorized float4 LDG.E.128 → STG.E.128 DMA from `d_forces` into the
/// active slot's pinned forces payload, with per-warp `__shfl_down_sync`
/// butterfly reduction of Σ Fᵢ² and warp-leader `atomicAdd` directly into
/// the slot's pinned `force_norm` field (offset 28). The field holds the
/// running sum-of-squares after this kernel; the sqrt post-pass converts
/// it to the L2 norm in-place.
///
/// Caller invariant: `force_norm` MUST be 0.0f before launch. The host
/// ZSTR Reaper resets it alongside `completion_fence` once it has read
/// and validated the slot.
extern "C"
__global__ void zstr_force_stage_f4_kernel(
    uint8_t* __restrict__       base_pinned,
    uint32_t                    inter_slot_stride,
    uint32_t                    force_offset_in_slot,
    uint32_t                    force_norm_offset_in_slot,
    const float4* __restrict__  src_forces,
    uint32_t                    n_floats,
    const uint32_t* __restrict__ d_active_slot
);

/// T11 — Single-thread <<<1,1>>> post-pass.  Reads the active slot's
/// `force_norm` (= Σ Fᵢ² as written by force_stage), writes `sqrtf`
/// back in place, converting the field to the L2 norm ‖F‖₂.  NaN
/// propagates verbatim — the host G29 Reaper traps non-finite values.
extern "C"
__global__ void zstr_force_norm_sqrt_kernel(
    uint8_t* __restrict__       base_pinned,
    uint32_t                    inter_slot_stride,
    uint32_t                    force_norm_offset_in_slot,
    const uint32_t* __restrict__ d_active_slot,
    /// P8 — pointer-stable mirror for the G26 governor's force guard
    /// (nullable). The ring slot rolls `step % 5`; the governor binds one
    /// address for the life of the captured graph.
    float* __restrict__         d_force_norm_mirror
);

}} // namespace prism_nhs::zstr

// ─── C-ABI capture-window launchers ─────────────────────────────────────────
// Called from Rust captured_pipeline.rs inside cuStreamBeginCapture to record
// ZSTR kernel nodes into the in-progress CUgraph.
//
// All functions return `cudaError_t` cast to `int`; 0 == cudaSuccess.
// They must be called on the telemetry_stream (non-blocking) AFTER the tile
// DMA cuMemcpyDtoHAsync and BEFORE the telemetry→MD JOIN event record.

#ifdef __cplusplus
extern "C" {
#endif

/// WP4 §A launcher: records the slot-authority kernel.  MUST be recorded
/// once per graph-body iteration, upstream of every other ZSTR node, on a
/// stream that has a happens-before edge into the telemetry stream.
///
/// `d_step_counter`            : u64 device pointer (§1 device clock).
/// `n_slots`                   : ZstrRing::N_SLOTS.
/// `base_pinned`               : slot-0 header start.
/// `inter_slot_stride`         : ZstrRing::frame_size.
/// `d_active_slot`             : per-stream `uint32_t*` (ring control sector).
/// `n_atoms` / `dt`            : header stamp values (dt in PICOSECONDS).
/// `gear_id`                   : header stamp value (G26 chronometric gear).
/// `stream_id` / `schema_version` : header stamp values.
/// `backpressure_spin_limit`   : bounded wait iterations; 0 disables the
///                               wait (legacy overwrite semantics — never
///                               used by the production path).
/// `d_backpressure_error`      : per-stream status block (see
///                               PRISM_ZSTR_BP_OFF_* above); may be null.
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
                                          void*       stream);

/// WP4 launcher: per-stream-slot pos_stage (kernel reads *d_active_slot).
/// `base_pinned` is the slot-0 header start; `inter_slot_stride` is the
/// inter-slot byte distance (== ring frame_size); `pos_offset_in_slot` is
/// the byte offset from slot base to positions payload (== sizeof header).
int zstr_launch_pos_stage(void*       base_pinned,
                           uint32_t    inter_slot_stride,
                           uint32_t    pos_offset_in_slot,
                           const void* src_vram,
                           uint32_t    n_atoms,
                           const void* d_active_slot,
                           void*       stream);

/// WP4 launcher: per-stream-slot fence signal.  base_fence points at slot-0
/// completion_fence; kernel rolls slot via `*d_active_slot`.
int zstr_launch_fence_signal(void*    base_fence,
                              uint32_t inter_slot_stride,
                              const void* d_active_slot,
                              void*    stream);

/// DEPRECATED (WP4) host helper, preserved and reachable.  Stream-ordered
/// device write of an explicit slot index into a PER-STREAM `d_active_slot`
/// cell.  Superseded by the in-graph slot-authority kernel, which is the
/// only path that also stamps the frame header; calling this without the
/// in-graph kernel produces slot claims with unstamped headers.  Retained
/// for no-unroll debug fallbacks and single-shot harnesses.
int prism_zstr_set_active_slot_ptr(void* d_active_slot, uint32_t slot, void* stream);

/// T11 — Records the force-stage kernel onto `stream` inside the
/// active capture window.  DMA copy + warp-shuffle reduce + atomic-add
/// into pinned `force_norm`.  Caller pairs this with
/// `zstr_launch_force_norm_sqrt` on the same stream for the in-place
/// sqrt post-pass.  Insert AFTER ASC atomic-add and BEFORE
/// `zstr_launch_fence_signal`.
int zstr_launch_force_stage(void*       base_pinned,
                             uint32_t    inter_slot_stride,
                             uint32_t    force_offset_in_slot,
                             uint32_t    force_norm_offset_in_slot,
                             const void* src_d_forces,
                             uint32_t    n_atoms,
                             const void* d_active_slot,
                             void*       stream);

/// T11 — Records the single-thread sqrt post-pass that finalises
/// `force_norm` to the L2 norm.  Insert immediately after
/// `zstr_launch_force_stage` and before `zstr_launch_fence_signal`.
/// P8: `d_force_norm_mirror` (nullable) receives the same L2 norm at a
/// POINTER-STABLE address, because the G26 governor binds one pointer for the
/// life of the captured graph and the ring slot rolls `step % 5`.
int zstr_launch_force_norm_sqrt(void*    base_pinned,
                                 uint32_t inter_slot_stride,
                                 uint32_t force_norm_offset_in_slot,
                                 const void* d_active_slot,
                                 void*    stream,
                                 void*    d_force_norm_mirror);

/// **M1.2.18.5** — Records the single-thread audit-field stage kernel
/// that snapshots the Hamiltonian audit fields into the active ZSTR slot:
///   slot[external_work_offset_in_slot]   ← *adj.d_external_work (offset 32, 8 B)
///   slot[potential_energy_offset_in_slot] ← adj.d_potential_energy (offset 40, 8 B)
/// WP4 additionally stamps `adjudication_code` (offset 12) from
/// `adj + 52` — this kernel is the only ZSTR node captured downstream of
/// the Adjudicator, so it is the sole owner of that field.
/// Captured downstream of the SFA (so V_t and W_ext are settled).  `adj`
/// may be null in legacy / test fixtures — kernel writes 0.0 to both.
int zstr_launch_stage_audit(void*       base_pinned,
                             uint32_t    inter_slot_stride,
                             uint32_t    external_work_offset_in_slot,
                             uint32_t    potential_energy_offset_in_slot,
                             uint32_t    adjudication_code_offset_in_slot,
                             const void* adj,
                             const void* d_active_slot,
                             void*       stream);

#ifdef __cplusplus
}
#endif
