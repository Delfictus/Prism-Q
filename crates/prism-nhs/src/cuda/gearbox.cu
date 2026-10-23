// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / G26 Chronometric Gearbox — implementation (Wave B.1)
// ═══════════════════════════════════════════════════════════════════════════

#include "gearbox.cuh"
#include "prism_physical_guard.cuh"
#include "adjudicator.cuh"  // CHUNK13_DIAG: PrismBranchTrace full definition
                            // for the traced predicate bridge variant below.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>  // Amendment 3.10 — kernel-level printf triage
#include <cassert>
#include <cstring> // GEAR-PROVENANCE-002 — memset of the host snapshot record
#include <limits>
#include <time.h>  // GEAR-PROVENANCE-002 — nanosleep throttle for the bounded wait

// ─── __constant__ gear table — 64 bytes, one Blackwell L1 const-cache line ──
//
// Initialised by `prism_gearbox_init_table_async`. The kernel reads via
// LDC (load constant) which broadcasts to all 32 lanes in a single
// instruction.  Static initialiser provides a safe pre-init value (Gear 1
// dt = 2.0 fs in slot 4) so a kernel launched before the host writes the
// real table observes a sensible default rather than zero.

// __constant__ static initialisers must be compile-time constants;
// CUDART_NAN_F is not constexpr (dynamic-init forbidden by nvcc), so the
// static placeholder uses safe Gear-1 dt for slot 0 of every gear and zero
// elsewhere.  The Rust-side `prism_gearbox_init_table_async` is invoked
// UNCONDITIONALLY in `CapturedAdjudicationPipeline::build` BEFORE any
// kernel launch, overwriting all 16 slots with the canonical layout
// (NaN sentinel for Gear 3 included).  So the static defaults below are
// only observable in unit tests that bypass `init_table_async`.
__constant__ float d_gearbox_table[PRISM_GEARBOX_TABLE_LEN] = {
    0.0005f, 0.0f, 0.0f, 0.0f,    // Gear 0 — 0.5 fs (placeholder)
    0.0020f, 0.0f, 0.0f, 0.0f,    // Gear 1 — 2.0 fs (placeholder)
    0.0040f, 0.0f, 0.0f, 0.0f,    // Gear 2 — 4.0 fs (placeholder)
    std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 0.0f, // Gear 3 abort sentinel; immutable in production
};

// ─── B.3 — 3×3 Constant Ratio Matrix ────────────────────────────────────────
//
// Gear-to-gear dt-ratio lookup table.  Indexed as [prev*3 + target].
// Pre-baked from the canonical gearbox dt values (0.5/2.0/4.0 fs).  Gear 3
// (abort) is OOB — the body 3 sub-graph fires the trap kernel without
// consulting this matrix.  Operator constraint: zero divisions in the
// rescale kernel — single LDC + MUL per thread.
//
// Static init values are correct for the canonical gearbox table; the
// host-side init helper exists for ABI symmetry with future runtime
// re-tuning (e.g., a different 3-gear ratio set on a non-HMR substrate).

__constant__ float d_rescale_ratios[9] = {
    1.0f,   4.0f,   8.0f,    // prev = 0 (0.5 fs) → target 0/1/2
    0.25f,  1.0f,   2.0f,    // prev = 1 (2.0 fs) → target 0/1/2
    0.125f, 0.5f,   1.0f,    // prev = 2 (4.0 fs) → target 0/1/2
};

// Governor policy, bindings and dwell state are per-stream allocations (V2).

// ─── adj→adjudication_code / d_dt offsets ───────────────────────────────────
//
// Pinned by the C-side static_assert in adjudicator.cuh:
//   adjudication_code  offset 52  (u32)
//   d_dt               offset 112 (*mut f32)
// Reading via offset arithmetic on a `const uint8_t*` keeps this TU
// independent of the full FFI struct definition (operator §M2 / Anti-
// Greenfield: don't drag SO(3) tile types into a kernel that doesn't
// need them).

__device__ __forceinline__ uint32_t prism_gearbox_load_adj_code(
    const InterferometricAdjudicatorFfi* adj
) {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(adj);
    return *reinterpret_cast<const uint32_t*>(base + 52);
}

__device__ __forceinline__ float* prism_gearbox_load_adj_d_dt(
    const InterferometricAdjudicatorFfi* adj
) {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(adj);
    // d_dt is a *mut f32; we load the 8-byte pointer value.
    // M1.2.17: moved from offset 112 → 120 (potential_energy now at 112).
    return *reinterpret_cast<float* const*>(base + 120);
}

/// M1.2.17 — load adj->potential_energy (f64 VALUE at offset 112).
/// Read by the SFA stability-fuse logic to compute the rolling-window
/// drift |V_t − V_{t-1}| / |V_{t-1}|.
__device__ __forceinline__ double prism_gearbox_load_adj_v_curr(
    const InterferometricAdjudicatorFfi* adj
) {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(adj);
    return *reinterpret_cast<const double*>(base + 112);
}

// ─── PointerSwap kernel — Stateful Finite Automaton ────────────────────────
//
// Single-thread.  All fields read/written here live on device:
//   adj->adjudication_code  ←  Adjudicator step kernel
//   adj->d_dt              ←  pre-capture wire-up (= &d_protocol->dt)
//   cruise->{counter,...}  ←  ChronometricStateTensor (this kernel + reset)
//
// State transitions:
//   code = 2  →  Gear 3 (Violation; abort).  Counter unchanged so the
//                cruise can be inspected post-mortem.
//   code = 1  →  Gear 0 (Burst); cruise.counter = 0;
//                cruise.last_burst_frame = current_frame.
//   code = 0  →  cruise.counter += 1; gear = (counter < THRESH) ? 1 : 2.

extern "C"
__global__ void prism_gearbox_pointer_swap_kernel(
    const InterferometricAdjudicatorFfi* __restrict__ adj,
    ChronometricStateTensor*             __restrict__ cruise,
    uint32_t                                          current_frame
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint32_t code = prism_gearbox_load_adj_code(adj);

    uint32_t counter           = cruise->counter;
    uint32_t last_burst_frame  = cruise->last_burst_frame;
    uint32_t target_gear;

    if (code == 2u) {
        // Hard-trap path.  Wave B.2's SWITCH routes Gear 3 to the PTX
        // trap kernel; even if that wiring is bypassed, writing a NaN dt
        // through *(adj->d_dt) gives integrators a non-finite step that
        // their existing FP-exception guards catch.
        target_gear = 3u;
    } else if (code == 1u) {
        target_gear       = 0u;
        counter           = 0u;
        last_burst_frame  = current_frame;
    } else {
        // code == 0 (Equilibrium / Prune) — cruise hysteresis.
        // Saturating increment so a long quiet run doesn't wrap u32.
        counter           = (counter < 0xffffffffu) ? counter + 1u : counter;
        target_gear       = (counter < PRISM_GEARBOX_CRUISE_THRESHOLD) ? 1u : 2u;
    }

    // B.2 — capture the OLD current_gear into previous_gear so the
    // symplectic ratio kernel can compute λ = dt_new / dt_old without
    // a separate scratch buffer.  Read happens BEFORE the write below.
    const uint32_t prev_gear  = cruise->current_gear;
    cruise->previous_gear     = prev_gear;
    cruise->counter           = counter;
    cruise->last_burst_frame  = last_burst_frame;
    cruise->current_gear      = target_gear;

    // Hardware write to ProtocolState.dt (or whatever address d_dt
    // points to).  d_dt is null in Wave A pipelines (Pre-Flight);
    // a guard avoids a segfault if the pipeline forgot to wire it.
    float* dt_target = prism_gearbox_load_adj_d_dt(adj);
    if (dt_target != nullptr) {
        const uint32_t slot = target_gear * PRISM_GEARBOX_FLOATS_PER_GEAR;
        *dt_target = d_gearbox_table[slot];
    }
}

// ─── Host-side init: 16-float table → __constant__ d_gearbox_table ─────────

extern "C"
int prism_gearbox_init_table_async(
    const float* host_table,
    void*        stream)
{
    if (host_table == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t rc = cudaMemcpyToSymbolAsync(
        d_gearbox_table,
        host_table,
        sizeof(float) * PRISM_GEARBOX_TABLE_LEN,
        0,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream)
    );
    return static_cast<int>(rc);
}

// ─── Host launcher ──────────────────────────────────────────────────────────

extern "C"
int prism_gearbox_launch_pointer_swap(
    const InterferometricAdjudicatorFfi* adj,
    ChronometricStateTensor*             cruise,
    uint32_t                             current_frame,
    void*                                stream)
{
    if (adj == nullptr || cruise == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    prism_gearbox_pointer_swap_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(adj, cruise, current_frame);
    return static_cast<int>(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════
// GEAR-PROVENANCE-002 — device → host gear measurement
// ════════════════════════════════════════════════════════════════════
//
// The gearbox's whole decision record lived on the device: `cruise->
// current_gear` never left the GPU, so every host-side `gear_id` in the
// tree was a build-time literal masquerading as a measurement.  This block
// is the missing read path.  It is deliberately OUTSIDE any capture window
// (host-resident copies are capture-illegal) and its only wait is bounded.

namespace {

/// Microseconds on CLOCK_MONOTONIC.  Used so the reported wait is a CLOCK
/// READING, not a count of nominal sleep quanta: `nanosleep` may oversleep by
/// an unbounded amount under load, so accumulating `POLL_US` per iteration
/// would publish a fabricated duration in an evidence record and would also
/// let the real wait exceed the stated budget.
__host__ uint64_t prism_gearbox_monotonic_us()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0ull;
    return static_cast<uint64_t>(ts.tv_sec) * 1000000ull
         + static_cast<uint64_t>(ts.tv_nsec) / 1000ull;
}

/// Bounded drain wait on `s`.  Records a disable-timing event and polls it
/// with a nanosleep throttle until completion or `budget_us` of MEASURED wall
/// time expires.  Never spins; never waits without a bound.
///
/// Returns cudaSuccess when the stream drained, PRISM_GEARBOX_RC_TIMEOUT on
/// expiry, or the failing cudaError_t.  `waited_us_out` is always written and
/// is always a measured elapsed time.
///
/// Destroying a still-pending event is legal: the driver defers resource
/// release until the event completes, so the timeout path leaks nothing and
/// leaves no host-visible write outstanding.
__host__ int prism_gearbox_bounded_drain(
    cudaStream_t s,
    uint64_t     budget_us,
    uint64_t*    waited_us_out)
{
    if (waited_us_out != nullptr) *waited_us_out = 0ull;

    cudaEvent_t ev = nullptr;
    cudaError_t rc = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    if (rc != cudaSuccess) return static_cast<int>(rc);

    rc = cudaEventRecord(ev, s);
    if (rc != cudaSuccess) {
        cudaEventDestroy(ev);
        return static_cast<int>(rc);
    }

    const uint64_t t0        = prism_gearbox_monotonic_us();
    uint64_t       waited_us = 0ull;
    cudaError_t    q         = cudaEventQuery(ev);
    while (q == cudaErrorNotReady && waited_us < budget_us) {
        struct timespec ts;
        ts.tv_sec  = 0;
        ts.tv_nsec = static_cast<long>(PRISM_GEARBOX_SNAPSHOT_POLL_US) * 1000L;
        nanosleep(&ts, nullptr);
        const uint64_t now = prism_gearbox_monotonic_us();
        waited_us = (now >= t0) ? (now - t0) : 0ull;
        q = cudaEventQuery(ev);
    }

    cudaEventDestroy(ev);
    if (waited_us_out != nullptr) *waited_us_out = waited_us;

    if (q == cudaSuccess)       return static_cast<int>(cudaSuccess);
    if (q == cudaErrorNotReady) return PRISM_GEARBOX_RC_TIMEOUT;
    return static_cast<int>(q);
}

/// Initialise `out` to the explicit no-measurement state.  Called before any
/// device access so that EVERY return path — including early refusals — hands
/// back sentinels rather than plausible zeros.
__host__ void prism_gearbox_snapshot_reset(
    PrismGearboxHostSnapshot* out,
    uint32_t                  status)
{
    memset(out, 0, sizeof(*out));
    out->schema_version   = PRISM_GEARBOX_SNAPSHOT_SCHEMA_VERSION;
    out->status           = status;
    out->current_gear     = PRISM_GEARBOX_GEAR_UNMEASURED;
    out->previous_gear    = PRISM_GEARBOX_GEAR_UNMEASURED;
    out->dt_ps_applied    = 0.0f;
    out->dt_valid         = 0u;
}

}  // anonymous namespace

extern "C"
int prism_gearbox_snapshot_state(
    const ChronometricStateTensor*  cruise,
    const float*                    d_dt,
    void*                           stream,
    uint32_t                        timeout_ms,
    struct PrismGearboxHostSnapshot* out)
{
    if (out == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    prism_gearbox_snapshot_reset(out, PRISM_GEARBOX_SNAPSHOT_NULL_ARG);
    if (cruise == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // ── 1. Capture guard ──────────────────────────────────────────────
    // A host-resident D2H copy inside a capture window corrupts the graph.
    // Any non-success query result is treated as "capturing" (fail-closed):
    // the legacy default stream reports cudaErrorStreamCaptureImplicit while
    // another stream in the same context is capturing.
    if (s != nullptr) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        const cudaError_t crc = cudaStreamIsCapturing(s, &cs);
        if (crc != cudaSuccess || cs != cudaStreamCaptureStatusNone) {
            prism_gearbox_snapshot_reset(out, PRISM_GEARBOX_SNAPSHOT_CAPTURE_ACTIVE);
            return static_cast<int>(crc != cudaSuccess
                                    ? crc
                                    : cudaErrorStreamCaptureUnsupported);
        }
    }

    // ── 2. Bounded drain wait ─────────────────────────────────────────
    const uint32_t budget_ms = (timeout_ms == 0u)
        ? PRISM_GEARBOX_SNAPSHOT_DEFAULT_TIMEOUT_MS
        : timeout_ms;
    uint64_t  waited_us = 0ull;
    const int drain_rc  = prism_gearbox_bounded_drain(
        s, static_cast<uint64_t>(budget_ms) * 1000ull, &waited_us);
    if (drain_rc == PRISM_GEARBOX_RC_TIMEOUT) {
        prism_gearbox_snapshot_reset(out, PRISM_GEARBOX_SNAPSHOT_TIMEOUT);
        out->waited_us = waited_us;
        return PRISM_GEARBOX_RC_TIMEOUT;
    }
    if (drain_rc != static_cast<int>(cudaSuccess)) {
        prism_gearbox_snapshot_reset(out, PRISM_GEARBOX_SNAPSHOT_EVENT_FAILED);
        out->waited_us = waited_us;
        return drain_rc;
    }

    // ── 3. Synchronous D2H on a drained stream ────────────────────────
    ChronometricStateTensor host_cruise{};
    cudaError_t rc = cudaMemcpy(&host_cruise, cruise,
                                sizeof(ChronometricStateTensor),
                                cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) {
        prism_gearbox_snapshot_reset(out, PRISM_GEARBOX_SNAPSHOT_COPY_FAILED);
        out->waited_us = waited_us;
        return static_cast<int>(rc);
    }

    float    dt_ps    = 0.0f;
    uint32_t dt_valid = 0u;
    if (d_dt != nullptr) {
        rc = cudaMemcpy(&dt_ps, d_dt, sizeof(float), cudaMemcpyDeviceToHost);
        if (rc != cudaSuccess) {
            // The gear itself IS measured; only the applied dt is not.
            // Report the gear and leave dt explicitly invalid rather than
            // discarding a good measurement or inventing a timestep.
            dt_ps    = 0.0f;
            dt_valid = 0u;
        } else {
            dt_valid = 1u;
        }
    }

    out->schema_version  = PRISM_GEARBOX_SNAPSHOT_SCHEMA_VERSION;
    out->status          = PRISM_GEARBOX_SNAPSHOT_OK;
    out->counter         = host_cruise.counter;
    out->last_burst_frame= host_cruise.last_burst_frame;
    out->current_gear    = host_cruise.current_gear;
    out->previous_gear   = host_cruise.previous_gear;
    out->dt_ps_applied   = dt_ps;
    out->dt_valid        = dt_valid;
    out->v_prev          = host_cruise.v_prev;
    out->waited_us       = waited_us;
    return static_cast<int>(cudaSuccess);
}

// ─── B.3.2 — SFA-only kernel (Logic/Action Bifurcation) ────────────────────
//
// Runs the same stateful finite automaton as prism_gearbox_pointer_swap_kernel
// but does NOT write *(adj->d_dt).  The dt-write side-effect is owned by the
// SWITCH body's apply-fixed-dt kernel.  This decoupling lets the Blackwell
// scheduler hide the SFA's logic-only cycles behind the integrator's global-
// memory writes — the operator-mandated path to >95% SM utilization.

extern "C"
__global__ void prism_gearbox_sfa_kernel_v2(
    const InterferometricAdjudicatorFfi* __restrict__ adj,
    ChronometricStateTensor*             __restrict__ cruise,
    PrismGearGovernorDeviceV2*                         governor
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (!governor || governor->abi_version != 2u || governor->byte_size != sizeof(*governor) ||
        !governor->bindings.physical_epoch) { asm volatile("trap;"); return; }
    const uint64_t physical_epoch = *governor->bindings.physical_epoch;
    const uint32_t current_frame = static_cast<uint32_t>(physical_epoch);
    const uint32_t code = prism_gearbox_load_adj_code(adj);

    uint32_t counter           = cruise->counter;
    uint32_t last_burst_frame  = cruise->last_burst_frame;
    uint32_t target_gear;

    if (code == 2u) {
        target_gear = 3u;
    } else if (code == 1u) {
        target_gear       = 0u;
        counter           = 0u;
        last_burst_frame  = current_frame;
    } else {
        // WP C — the cruise hysteresis is now policy-resolvable rather than a
        // source literal.  This governs the REQUESTER (when the KL automaton
        // asks for the campaign gear); the governor below still has to
        // authorize it.  0 reads as "use the documented default", never as
        // "no hysteresis".
        const uint32_t cruise_threshold =
            (governor->policy.cruise_threshold_steps == 0u)
                ? PRISM_GEAR_DEFAULT_CRUISE_THRESHOLD_STEPS
                : governor->policy.cruise_threshold_steps;
        counter     = (counter < 0xffffffffu) ? counter + 1u : counter;
        target_gear = (counter < cruise_threshold) ? 1u : 2u;
    }

    if (governor->bindings.reserved) {
        auto* guard = reinterpret_cast<PrismPhysicalGuardV1*>(governor->bindings.reserved);
        auto& s = guard->snapshot;
        if (s.generation != governor->allocation_generation || s.source != governor->source_stream_id ||
            s.physical_eval_epoch != physical_epoch || !physical_epoch || s.science_epoch >= physical_epoch || code > 2u ||
            (adj->gear_override != 0xffu && adj->gear_override > PRISM_GEAR_ABORT)) {
            s.status = PRISM_GUARD_EPOCH_MISMATCH; asm volatile("trap;"); return;
        }
        s.science_epoch = physical_epoch; ++s.science_publications;
        s.science_status = PRISM_SCIENCE_PUBLISHED;
        s.science_code = code; s.science_request = target_gear;
        s.science_override = adj->gear_override;
        if (adj->momentum_violation_flag) s.science_code = 2u;
        s.science_age_physical_steps = 0;
        cruise->counter = counter; cruise->last_burst_frame = last_burst_frame;
        // The completed-chunk observer publishes intent. The physical guard
        // owns velocity transforms and dt writes inside the resident body.
        return;
    }

    // ── First-Law Hamiltonian Stability Fuse (Zero-Trust §3.2) ──
    // Read V_t from adj.potential_energy (offset 112) and V_{t-1}
    // from cruise.v_prev.  Read W_ext from *adj.d_external_work
    // (offset 128, POINTER) — the integrated non-conservative energy
    // injected by UV kicks, ASC, force/velocity clamps, etc., over the
    // chunk window.  The pointer holds an F2-pool address; the
    // captured graph emits cuMemsetD8Async at the head of every replay
    // so the deref'd value is fresh per chunk.
    //
    // Operator M1.2.18.5 §2 — Corrected Hamiltonian Audit:
    //   Drift = |V_t − (V_{t-1} + W_ext)| / |V_{t-1} + W_ext|
    //
    // Subtracting W_ext means the fuse fires only on numerical
    // instability, not on intentional perturbation.  The 1% threshold
    // catches drift that's NOT explained by tracked external work.
    //
    // First-frame guard: when v_prev == 0.0 (sentinel), skip the
    // drift check.  Null d_external_work pointer treated as W_ext = 0.
    const double v_curr = prism_gearbox_load_adj_v_curr(adj);
    const double v_prev = cruise->v_prev;
    // M1.2.18.5 — Production W_ext pointer-deref restored after diag2
    // confirmed the deref is NOT the source of the V2-launch
    // MISALIGNED_ADDRESS fault.  Sanitizer interrogation (Option B) will
    // pinpoint the actual faulting kernel.
    const uint8_t* adj_base = reinterpret_cast<const uint8_t*>(adj);
    const double* d_w_ext_ptr =
        *reinterpret_cast<const double* const*>(adj_base + 128);
    const double w_ext = (d_w_ext_ptr != nullptr) ? *d_w_ext_ptr : 0.0;
    const bool   first_frame = (v_prev == 0.0);
    // WP C — the drift is now MEASURED and handed to the governor rather than
    // being consumed by an inline `> 0.01 ? gear 3` test.  Gate 1 of the
    // governor applies the identical abort bound (policy `drift_abort`,
    // default 0.01 — the same source literal), and gates 4/5 add the
    // graduated downshift bands the inline test could not express.
    // `drift_measured == 0` on the first frame is an explicit UNMEASURED
    // marker, never a substituted 0.0.
    float drift_f32     = 0.0f;
    uint32_t drift_meas = 0u;
    if (!first_frame) {
        const double expected = v_prev + w_ext;
        const double abs_exp  = (expected < 0.0) ? -expected : expected;
        const double diff     = v_curr - expected;
        const double abs_diff = (diff < 0.0) ? -diff : diff;
        // Avoid division by zero when |V_{t-1} + W_ext| ≈ 0 (rare).
        const double drift    = (abs_exp > 1.0e-12)
                                ? (abs_diff / abs_exp)
                                : 0.0;
        drift_f32  = static_cast<float>(drift);
        drift_meas = 1u;
    }
    // isnan check: NaN != NaN.  Reject Inf as well.
    const bool nan_v =
        !(v_curr == v_curr) || (v_curr > 1.0e300) || (v_curr < -1.0e300);

    // ── WP C — PHYSICAL STABILITY GOVERNOR ───────────────────────────────
    //
    // `target_gear` above is the scientific-resolution REQUEST: it is derived
    // from the Adjudicator's KL burst statistic and the cruise counter, which
    // between them know nothing about numerical stability.  The governor
    // publishes the CEILING, and the authorized gear is the more conservative
    // of the two.  See the architecture block in gearbox.cuh.
    //
    // The governor is consulted UNCONDITIONALLY on this path.  When it has
    // not been armed the ceiling pins to the documented unknown-topology cap
    // (Gear 1 / 2.0 fs) and `binding_gate` records
    // PRISM_GEAR_GATE_GOVERNOR_UNARMED, so an unarmed run can never reach the
    // 4 fs campaign gear on an unverified topology — the exact failure the
    // old `counter >= 500` predicate allowed.
    const PrismGearGovernorPolicy* pol = &governor->policy;

    const auto* fw = governor->bindings.force_window;
    const auto* cw = governor->bindings.constraint_window;
    governor->force_status = prism_force_window_status(fw, governor->source_stream_id,
        governor->allocation_generation, physical_epoch);
    governor->constraint_status = prism_constraint_window_status(cw, governor->source_stream_id,
        governor->allocation_generation, physical_epoch);
    const uint32_t force_meas = governor->force_status == PRISM_WINDOW_COMPLETE;
    const float force_norm = force_meas ? fw->force_norm : CUDART_NAN_F;
    const uint32_t cons_meas = governor->constraint_status == PRISM_WINDOW_COMPLETE;
    const uint32_t cons_fail = cons_meas ? cw->fatal_failure_count : PRISM_GEAR_CONSTRAINT_UNMEASURED;
    governor->observed_force_epoch = force_meas ? fw->window_epoch : 0;
    governor->observed_constraint_epoch = cons_meas ? cw->window_epoch : 0;
    governor->last_physical_epoch = physical_epoch;
    governor->decision_epoch += 1;

    // adj.momentum_violation_flag lives at offset 144 (pinned by the
    // static_assert in adjudicator.cuh).
    const uint32_t momentum_flag =
        *reinterpret_cast<const uint32_t*>(adj_base + 144);

    PrismGearGuardObservation obs;
    obs.adjudication_code       = code;
    obs.momentum_violation      = momentum_flag;
    obs.nonfinite_energy        = (nan_v || physical_epoch > UINT32_MAX) ? 1u : 0u;
    obs.drift_measured          = drift_meas;
    obs.drift                   = drift_f32;
    obs.force_measured          = force_meas;
    obs.force_norm              = force_norm;
    obs.constraint_measured     = cons_meas;
    obs.constraint_failures     = cons_fail;
    obs.current_gear            = cruise->current_gear;
    obs.requested_gear          = target_gear;
    obs.stable_dwell_steps      = governor->state.stable_dwell_steps;
    obs.steps_since_instability = governor->state.steps_since_instability;
    obs.frame                   = current_frame;
    obs._reserved[0]            = 0u;
    obs._reserved[1]            = 0u;

    PrismGearDecision decision;
    prism_gearbox_governor_decide(
        pol, &obs, &governor->state.salience, &decision);

    const uint32_t authorized_gear = decision.authorized_gear;

    // Capture OLD current_gear into previous_gear so the body's
    // rescale kernel can read it for the symplectic ratio compute.
    const uint32_t prev_gear  = cruise->current_gear;
    cruise->previous_gear     = prev_gear;
    cruise->counter           = counter;
    cruise->last_burst_frame  = last_burst_frame;
    cruise->current_gear      = authorized_gear;
    // M1.2.17 — store V_t into v_prev so the next launch's SFA reads
    // the correct V_{t-1}.  (Captured-graph dependency edge: SFA runs
    // AFTER the energy monitor's window-update, so adj.potential_energy
    // already holds V_t.)
    cruise->v_prev            = v_curr;
    // No dt write — that's the SWITCH body's job.

    // ── Publish the decision record (device-resident; host reads it via
    //    prism_gearbox_governor_snapshot) ─────────────────────────────────
    PrismGearGovernorState* gs = &governor->state;
    gs->schema_version               = PRISM_GEAR_GOVERNOR_SCHEMA_VERSION;
    gs->decision_epoch               = gs->decision_epoch + 1u;
    gs->authorized_gear              = authorized_gear;
    gs->ceiling_gear                 = decision.ceiling_gear;
    gs->requested_gear               = target_gear;
    gs->binding_gate                 = decision.binding_gate;
    gs->gate_verdicts                = decision.gate_verdicts;
    gs->stable_dwell_steps           = decision.next_stable_dwell_steps;
    gs->steps_since_instability      = decision.next_steps_since_instability;
    gs->downshift_latency_steps      = decision.downshift_latency_steps;
    // Observed values are emitted with explicit UNMEASURED sentinels; an
    // absent measurement is NEVER written as a plausible number.
    gs->observed_drift               = drift_meas ? drift_f32 : CUDART_NAN_F;
    gs->observed_force_norm          = force_meas ? force_norm : CUDART_NAN_F;
    gs->observed_constraint_failures = cons_meas ? cons_fail
                                                 : PRISM_GEAR_CONSTRAINT_UNMEASURED;
    gs->emergency_flags              = decision.emergency_flags;
    gs->governor_armed               = pol->armed;
    gs->policy_schema_version        = pol->schema_version;
    gs->policy_digest                = pol->policy_digest;
    if (authorized_gear < prev_gear) {
        gs->last_downshift_epoch = gs->decision_epoch;
    } else if (authorized_gear > prev_gear) {
        gs->last_upshift_epoch = gs->decision_epoch;
    }
}

extern "C" int prism_gearbox_launch_sfa_v2(
    const InterferometricAdjudicatorFfi* adj, ChronometricStateTensor* cruise,
    PrismGearGovernorDeviceV2* governor, void* stream) {
    if (!adj || !cruise || !governor || !stream) return static_cast<int>(cudaErrorInvalidValue);
    prism_gearbox_sfa_kernel_v2<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(adj, cruise, governor);
    return static_cast<int>(cudaGetLastError());
}
// The singleton entry point cannot provide stream ownership and is retired.
extern "C" int prism_gearbox_launch_sfa(
    const InterferometricAdjudicatorFfi*, ChronometricStateTensor*, uint32_t, void*) {
    return static_cast<int>(cudaErrorNotSupported);
}

// ════════════════════════════════════════════════════════════════════
// B.2 — Symplectic Velocity Rescale (T12.2)
// ════════════════════════════════════════════════════════════════════
//
// Pure momentum-scaling kernel.  Each thread multiplies one f32 of the
// AoS velocity buffer by `ratio`.  The block size is 256 → each warp
// reads 32 contiguous f32 = 128 bytes → ptxas emits LDG.E.128 + STG.E.128
// (the operator-mandated vectorised path).  No position/force touch.
//
// The float4-aligned vectorised form is documented in the header but
// the runtime kernel is the simpler scalar-per-thread variant whose
// generated PTX matches LDG.E.128 by warp-level coalescing — confirmed
// by ptxas -arch=sm_120 -O3 inspection of the compiled archive.

extern "C"
__global__ void prism_gearbox_velocity_rescale_kernel(
    float*    __restrict__ d_velocities,
    uint32_t               n_floats,
    float                  ratio
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_floats) return;
    // Pure read-modify-write; no positions, no forces.
    d_velocities[tid] = d_velocities[tid] * ratio;
}

extern "C"
int prism_gearbox_launch_velocity_rescale(
    float*    d_velocities,
    uint32_t  n_floats,
    float     ratio,
    void*     stream)
{
    if (d_velocities == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    if (n_floats == 0u)          return static_cast<int>(cudaSuccess);

    constexpr uint32_t BLOCK = 256u;
    const     uint32_t grid  = (n_floats + BLOCK - 1u) / BLOCK;
    prism_gearbox_velocity_rescale_kernel<<<grid, BLOCK, 0,
        static_cast<cudaStream_t>(stream)>>>(d_velocities, n_floats, ratio);
    return static_cast<int>(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════
// B.2 — Berendsen Weak-Coupling Guard (T12.3)
// ════════════════════════════════════════════════════════════════════
//
// Standard Berendsen thermostat:
//     λ = sqrt(1 + (Δt/τ_T)·(T₀/T − 1))
// Applied for ONE FRAME on Gear-0 entry to absorb the kinetic-energy
// shock from the high-energy burst that triggered the downshift.
//
// Defensive epsilon clamp: if T → 0 (sample collapse) or τ → 0
// (operator misconfigured), the argument to sqrt could go negative;
// we clamp to ε = 1e-6 so the result stays finite.  The host G29
// Reaper traps NaN/Inf force_norm reads as a backstop.

extern "C"
__global__ void prism_gearbox_berendsen_guard_kernel(
    float*       __restrict__ d_velocities,
    uint32_t                  n_floats,
    const float* __restrict__ d_current_temp,
    const float* __restrict__ d_dt,
    float                     target_temp_K,
    float                     tau_ps
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_floats) return;

    // Broadcast loads — every thread reads the same 4 bytes; L1
    // constant-cache hit after the first warp request.  Defensive
    // null-guard: if d_current_temp or d_dt is null (test fixture
    // forgot to wire them) we bail with no scaling so the velocities
    // stay physically meaningful.
    if (d_current_temp == nullptr || d_dt == nullptr) return;

    const float T   = *d_current_temp;
    const float dt  = *d_dt;

    // Argument-clamp keeps sqrt finite even on degenerate input.
    const float arg = 1.0f + (dt / tau_ps) * (target_temp_K / T - 1.0f);
    const float lambda = sqrtf(fmaxf(arg, 1.0e-6f));

    d_velocities[tid] = d_velocities[tid] * lambda;
}

extern "C"
int prism_gearbox_launch_berendsen_guard(
    float*       d_velocities,
    uint32_t     n_floats,
    const float* d_current_temp,
    const float* d_dt,
    float        target_temp_K,
    float        tau_ps,
    void*        stream)
{
    if (d_velocities == nullptr || d_current_temp == nullptr || d_dt == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_floats == 0u) return static_cast<int>(cudaSuccess);

    constexpr uint32_t BLOCK = 256u;
    const     uint32_t grid  = (n_floats + BLOCK - 1u) / BLOCK;
    prism_gearbox_berendsen_guard_kernel<<<grid, BLOCK, 0,
        static_cast<cudaStream_t>(stream)>>>(
        d_velocities, n_floats,
        d_current_temp, d_dt,
        target_temp_K, tau_ps);
    return static_cast<int>(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════
// B.2 — 4-way Predicate Bridge (T12.4)
// ════════════════════════════════════════════════════════════════════
//
// Trivial 1-thread kernel: read `cruise->current_gear` and forward
// to the conditional handle via cudaGraphSetConditional.  PointerSwap
// already executed the stateful finite automaton (code → gear) and
// wrote current_gear to the cruise tensor; this bridge just
// communicates that decision to the SWITCH node.
//
// Requires CUDA 12.4+ for cudaGraphSetConditional.  Production
// toolchain is CUDA 13.x so the path is always live.

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040

// Predicate-bridge with gear_override consult.
//
//   final_gear = (adj->gear_override != 0xFF)
//                ? adj->gear_override
//                : cruise->current_gear
//
// `adj->gear_override` is at offset 100 (B.3.2 home; Emergency
// Rectification 2026-05-02 reverted from the M1.2.18.5 attempt to
// relocate to 136).  Read via byte-offset arithmetic to keep this TU
// independent of the full FFI struct definition.  When the host
// writes 0..3 via cuMemcpyHtoDAsync to (adj + 100), the next captured-
// graph launch's SWITCH routes to the corresponding body — the
// Blackwell Hardware Interlock.
extern "C"
__global__ void prism_gearbox_predicate_bridge_kernel(
    cudaGraphConditionalHandle                          handle,
    const InterferometricAdjudicatorFfi* __restrict__   adj,
    const ChronometricStateTensor*       __restrict__   cruise
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (cruise == nullptr) return;

    // Default to cruise.current_gear if adj is null (test fixtures);
    // otherwise consult the u32 gear_override at offset 100 (B.3.2).
    //
    //   final_gear = (override == 0xFF) ? calculated : (override & 0x03)
    //
    // Branchless selection per operator §2 — predicate evaluates as a
    // single comparison + select; no warp divergence (single thread).
    uint32_t final_gear = cruise->current_gear;
    if (adj != nullptr) {
        const uint8_t* base = reinterpret_cast<const uint8_t*>(adj);
        const uint32_t override_val = *reinterpret_cast<const uint32_t*>(base + 100);
        if (override_val != 0xFFu) {
            final_gear = override_val & 0x03u;
        }
    }
    // SWITCH was forged with size=4; defensive 2-bit mask.
    cudaGraphSetConditional(handle, final_gear & 0x3u);
}

extern "C"
int prism_gearbox_launch_predicate_bridge(
    uint64_t                                handle_v,
    const InterferometricAdjudicatorFfi*    adj,
    const ChronometricStateTensor*          cruise,
    void*                                   stream)
{
    if (cruise == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    cudaGraphConditionalHandle handle =
        static_cast<cudaGraphConditionalHandle>(handle_v);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    prism_gearbox_predicate_bridge_kernel<<<1, 1, 0, s>>>(handle, adj, cruise);
    return static_cast<int>(cudaGetLastError());
}

// ─── G26 predicate-bridge kernel (TRACED variant) ─────────────────────
// CHUNK13_CAPTURED_GRAPH_LAUNCH_HANG diagnostic — mirrors the legacy
// kernel above plus atomic writes into a pinned-mapped `PrismBranchTrace`
// struct (definition in adjudicator.cuh). Pass `d_trace == nullptr` to
// disable trace writes — kernel is then byte-equivalent to the legacy
// bridge.
extern "C"
__global__ void prism_gearbox_predicate_bridge_kernel_traced(
    cudaGraphConditionalHandle                          handle,
    const InterferometricAdjudicatorFfi* __restrict__   adj,
    const ChronometricStateTensor*       __restrict__   cruise,
    PrismBranchTrace*                    __restrict__   d_trace)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (cruise == nullptr) return;
    uint32_t final_gear = cruise->current_gear;
    if (adj != nullptr) {
        const uint8_t* base = reinterpret_cast<const uint8_t*>(adj);
        const uint32_t override_val = *reinterpret_cast<const uint32_t*>(base + 100);
        if (override_val != 0xFFu) {
            final_gear = override_val & 0x03u;
        }
    }
    const uint32_t masked = final_gear & 0x3u;
    cudaGraphSetConditional(handle, masked);
    if (d_trace != nullptr) {
        atomicExch(&d_trace->g26_predicate_last, masked);
        atomicAdd(&d_trace->g26_branch_count[masked], 1u);
        atomicAdd(&d_trace->g26_bridge_invocations, 1u);
    }
}

extern "C"
int prism_gearbox_launch_predicate_bridge_traced(
    uint64_t                                handle_v,
    const InterferometricAdjudicatorFfi*    adj,
    const ChronometricStateTensor*          cruise,
    void*                                   stream,
    uint64_t                                branch_trace_dev)
{
    if (cruise == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    cudaGraphConditionalHandle handle =
        static_cast<cudaGraphConditionalHandle>(handle_v);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    PrismBranchTrace* d_trace =
        reinterpret_cast<PrismBranchTrace*>(branch_trace_dev);
    prism_gearbox_predicate_bridge_kernel_traced<<<1, 1, 0, s>>>(
        handle, adj, cruise, d_trace);
    return static_cast<int>(cudaGetLastError());
}

#else  // CUDART_VERSION < 12040

extern "C"
int prism_gearbox_launch_predicate_bridge(
    uint64_t                                /*handle_v*/,
    const InterferometricAdjudicatorFfi*    /*adj*/,
    const ChronometricStateTensor*          /*cruise*/,
    void*                                   /*stream*/)
{
    return static_cast<int>(cudaErrorNotSupported);
}

extern "C"
int prism_gearbox_launch_predicate_bridge_traced(
    uint64_t                                /*handle_v*/,
    const InterferometricAdjudicatorFfi*    /*adj*/,
    const ChronometricStateTensor*          /*cruise*/,
    void*                                   /*stream*/,
    uint64_t                                /*branch_trace_dev*/)
{
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12040

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040
__global__ void prism_gearbox_guard_bridge_kernel(cudaGraphConditionalHandle handle,
    const ChronometricStateTensor* cruise,PrismGearGovernorDeviceV2* g,PrismBranchTrace* trace) {
    if(threadIdx.x||blockIdx.x)return;
    auto* guard=reinterpret_cast<PrismPhysicalGuardV1*>(g->bindings.reserved);
    if(!guard || guard->snapshot.status!=PRISM_GUARD_COMPLETE){asm volatile("trap;");return;}
    const auto& s=guard->snapshot;
    const uint32_t gear=(g->policy.armed && prism_guard_requested_gear(s)==PRISM_GEAR_ABORT)
        ? PRISM_GEAR_ABORT : cruise->current_gear;
    if(gear>PRISM_GEAR_ABORT){asm volatile("trap;");return;}
    cudaGraphSetConditional(handle,gear);
    if(trace){atomicExch(&trace->g26_predicate_last,gear);atomicAdd(&trace->g26_branch_count[gear],1u);atomicAdd(&trace->g26_bridge_invocations,1u);}
}
#endif
extern "C" int prism_gearbox_launch_guard_bridge_v1(uint64_t handle,const ChronometricStateTensor* cruise,
    PrismGearGovernorDeviceV2* g,void* stream,uint64_t trace) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040
    if(!cruise||!g||!stream)return static_cast<int>(cudaErrorInvalidValue);
    prism_gearbox_guard_bridge_kernel<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<cudaGraphConditionalHandle>(handle),cruise,g,reinterpret_cast<PrismBranchTrace*>(trace));
    return static_cast<int>(cudaGetLastError());
#else
    return static_cast<int>(cudaErrorNotSupported);
#endif
}

// ════════════════════════════════════════════════════════════════════
// B.3-narrow — SWITCH body kernels
// ════════════════════════════════════════════════════════════════════

// ─── Rescale kernel (ratio matrix lookup, single multiply) ─────────────────

extern "C"
__global__ void prism_gearbox_rescale_kernel(
    float*                                      __restrict__ d_velocities,
    uint32_t                                                  n_floats,
    const ChronometricStateTensor*              __restrict__ cruise,
    uint32_t                                                  target_gear
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_floats) return;
    if (cruise == nullptr) return;

    // Single LDC for ratio (broadcast across the warp).
    const uint32_t prev_gear = cruise->previous_gear & 0x3u;
    const uint32_t tgt       = target_gear           & 0x3u;
    // Bounds: matrix only covers active gears 0..2.  Gear 3 (abort)
    // bodies do NOT call this kernel; the trap kernel fires directly.
    if (prev_gear >= 3u || tgt >= 3u) return;
    const float ratio = d_rescale_ratios[prev_gear * 3u + tgt];

    // Single LDG + MUL + STG per thread.  Block 256 ⇒ warp-coalesced
    // 128-byte memory transactions (LDG.E.128 / STG.E.128).
    d_velocities[tid] = d_velocities[tid] * ratio;
}

extern "C"
int prism_gearbox_launch_rescale(
    float*                              d_velocities,
    uint32_t                            n_floats,
    const ChronometricStateTensor*      cruise,
    uint32_t                            target_gear,
    void*                               stream)
{
    if (d_velocities == nullptr || cruise == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_floats == 0u) return static_cast<int>(cudaSuccess);
    constexpr uint32_t BLOCK = 256u;
    const     uint32_t grid  = (n_floats + BLOCK - 1u) / BLOCK;
    prism_gearbox_rescale_kernel<<<grid, BLOCK, 0,
        static_cast<cudaStream_t>(stream)>>>(
        d_velocities, n_floats, cruise, target_gear);
    return static_cast<int>(cudaGetLastError());
}

extern "C"
int prism_gearbox_init_rescale_ratios_async(void* stream)
{
    // Static initialiser at the top of this TU is canonical; this
    // function exists for ABI symmetry with future runtime re-tuning.
    // No-op on the canonical 0.5/2.0/4.0fs gearbox.
    (void)stream;
    return static_cast<int>(cudaSuccess);
}

// ─── Apply-fixed-dt kernel (per-body dt write) ─────────────────────────────

extern "C"
__global__ void prism_gearbox_apply_fixed_dt_kernel(
    const InterferometricAdjudicatorFfi*  __restrict__ adj,
    uint32_t                                            target_gear
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (adj == nullptr) return;
    // Read d_dt pointer via offset arithmetic — keeps this TU
    // independent of the full FFI struct definition (operator §M2).
    // M1.2.17 layout pivot: d_dt now at offset 120 (was 112).
    const uint8_t* base = reinterpret_cast<const uint8_t*>(adj);
    float* dt_target = *reinterpret_cast<float* const*>(base + 120);
    if (dt_target == nullptr) return;
    if (target_gear >= 4u) return;
    *dt_target = d_gearbox_table[target_gear * PRISM_GEARBOX_FLOATS_PER_GEAR];
}

extern "C"
int prism_gearbox_launch_apply_fixed_dt(
    const InterferometricAdjudicatorFfi* adj,
    uint32_t                             target_gear,
    void*                                stream)
{
    if (adj == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    prism_gearbox_apply_fixed_dt_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(adj, target_gear);
    return static_cast<int>(cudaGetLastError());
}

// ─── Trap kernel (PTX hardware-trap for Gear 3 abort) ──────────────────────

extern "C"
__global__ void prism_gearbox_trap_kernel() {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // PTX `trap` instruction.  Halts the GPU stream and surfaces
    // CUDA_ERROR_LAUNCH_FAILED to the host on the next sync.
    asm volatile("trap;");
}

extern "C"
int prism_gearbox_launch_trap(void* stream) {
    prism_gearbox_trap_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>();
    return static_cast<int>(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════
// B.3-narrow — SWITCH body populator (Blackwell Driver Probe target)
// ════════════════════════════════════════════════════════════════════
//
// Takes the four phGraph_out CUgraph handles returned by
// `prism_wire_g26_gearbox_ffi` and adds the body kernels as graph
// nodes.  If ANY of the four handles is null, returns
// cudaErrorInvalidValue early — the operator-mandated "Smoking Gun"
// signal that CUDA 13.x's runtime API does not populate phGraph_out
// for SWITCH-type conditional nodes (the same bug noted for IF-type
// at captured_pipeline.rs:1212-1216).  In that case the caller pivots
// to the `cudaGraphSetConditional` kernel-conditional pattern.
//
// Body topologies (per operator 2026-05-02 §1.1):
//   Body 0 (Gear 0 — Burst — 0.5 fs):
//     rescale(0) → [Berendsen guard if d_current_temp+d_dt non-null]
//                → apply_fixed_dt(0)
//   Body 1 (Gear 1 — Cruise — 2.0 fs):  rescale(1) → apply_fixed_dt(1)
//   Body 2 (Gear 2 — Sprint — 4.0 fs):  rescale(2) → apply_fixed_dt(2)
//   Body 3 (Gear 3 — Abort):            trap

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12060

namespace {

// Canonical kernel-node addition wrapper.  CUDA 13.x signature uses
// cudaGraphAddNode with a cudaGraphNodeParams struct (the runtime
// equivalent of cuGraphAddNode_v2).  We use the simpler
// cudaGraphAddKernelNode path which is stable across 12.4+/13.x.
__host__ cudaError_t add_kernel_node(
    cudaGraphNode_t*       out_node,
    cudaGraph_t            graph,
    const cudaGraphNode_t* deps,
    size_t                 num_deps,
    void*                  func,
    dim3                   grid,
    dim3                   block,
    void**                 args,
    size_t                 shared_bytes
) {
    cudaKernelNodeParams params{};
    params.func           = func;
    params.gridDim        = grid;
    params.blockDim       = block;
    params.sharedMemBytes = static_cast<unsigned int>(shared_bytes);
    params.kernelParams   = args;
    params.extra          = nullptr;
    return cudaGraphAddKernelNode(out_node, graph, deps, num_deps, &params);
}

}  // anonymous namespace

extern "C"
int prism_gearbox_populate_switch_bodies_ffi(
    cudaGraph_t*                          body_subgraphs,
    const InterferometricAdjudicatorFfi*  adj,
    float*                                d_velocities,
    uint32_t                              n_floats,
    const ChronometricStateTensor*        cruise,
    const float*                          d_current_temp,
    const float*                          d_dt,
    float                                 target_temp_K,
    float                                 tau_ps)
{
    if (body_subgraphs == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // ── Driver Probe: assert all four sub-graph handles are valid. ──
    // Operator-mandated "Smoking Gun" check (2026-05-02 §1.3): a null
    // handle here means CUDA 13.x's cudaGraphAddNode(CONDITIONAL)
    // does NOT populate phGraph_out for SWITCH-type, and we must
    // pivot to the kernel-conditional fallback path.
    for (int i = 0; i < 4; ++i) {
        if (body_subgraphs[i] == nullptr) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }

    constexpr uint32_t BLOCK = 256u;
    const     uint32_t rescale_grid =
        (n_floats == 0u) ? 1u : (n_floats + BLOCK - 1u) / BLOCK;

    cudaError_t rc = cudaSuccess;

    // ── Bodies 0, 1, 2: rescale → [Berendsen for 0] → apply_fixed_dt ──
    for (uint32_t gear = 0u; gear <= 2u; ++gear) {
        cudaGraph_t body = body_subgraphs[gear];

        cudaGraphNode_t rescale_node = nullptr;
        {
            void* args[] = {
                &d_velocities,
                const_cast<uint32_t*>(&n_floats),
                const_cast<ChronometricStateTensor**>(&cruise),
                &gear,
            };
            rc = add_kernel_node(
                &rescale_node, body,
                /*deps=*/   nullptr, /*num_deps=*/ 0,
                reinterpret_cast<void*>(prism_gearbox_rescale_kernel),
                dim3(rescale_grid, 1, 1), dim3(BLOCK, 1, 1),
                args, /*shared=*/ 0);
            if (rc != cudaSuccess) return static_cast<int>(rc);
        }

        cudaGraphNode_t last_node = rescale_node;

        // ── Berendsen guard (Gear 0 only — burst-shock dampener) ──
        if (gear == 0u && d_current_temp != nullptr && d_dt != nullptr) {
            cudaGraphNode_t berendsen_node = nullptr;
            void* args[] = {
                &d_velocities,
                const_cast<uint32_t*>(&n_floats),
                const_cast<float**>(&d_current_temp),
                const_cast<float**>(&d_dt),
                &target_temp_K,
                &tau_ps,
            };
            rc = add_kernel_node(
                &berendsen_node, body,
                &last_node, /*num_deps=*/ 1,
                reinterpret_cast<void*>(prism_gearbox_berendsen_guard_kernel),
                dim3(rescale_grid, 1, 1), dim3(BLOCK, 1, 1),
                args, /*shared=*/ 0);
            if (rc != cudaSuccess) return static_cast<int>(rc);
            last_node = berendsen_node;
        }

        // ── apply_fixed_dt — single-thread dt write ──
        cudaGraphNode_t apply_node = nullptr;
        {
            void* args[] = {
                const_cast<InterferometricAdjudicatorFfi**>(&adj),
                &gear,
            };
            rc = add_kernel_node(
                &apply_node, body,
                &last_node, /*num_deps=*/ 1,
                reinterpret_cast<void*>(prism_gearbox_apply_fixed_dt_kernel),
                dim3(1, 1, 1), dim3(1, 1, 1),
                args, /*shared=*/ 0);
            if (rc != cudaSuccess) return static_cast<int>(rc);
        }
    }

    // ── Body 3: trap kernel ───────────────────────────────────────
    {
        cudaGraph_t body = body_subgraphs[3];
        cudaGraphNode_t trap_node = nullptr;
        rc = add_kernel_node(
            &trap_node, body,
            /*deps=*/ nullptr, /*num_deps=*/ 0,
            reinterpret_cast<void*>(prism_gearbox_trap_kernel),
            dim3(1, 1, 1), dim3(1, 1, 1),
            /*args=*/ nullptr, /*shared=*/ 0);
        if (rc != cudaSuccess) return static_cast<int>(rc);
    }

    return static_cast<int>(cudaSuccess);
}

__global__ void prism_gearbox_guard_observe_kernel(const PrismGearGovernorDeviceV2* g,uint32_t gear) {
    if(threadIdx.x||blockIdx.x)return;
    const auto* guard=reinterpret_cast<const PrismPhysicalGuardV1*>(g->bindings.reserved);
    if(!guard||guard->snapshot.status!=PRISM_GUARD_COMPLETE||guard->snapshot.applied_gear!=gear){asm volatile("trap;");}
}
extern "C" int prism_gearbox_populate_guard_bodies_v1(cudaGraph_t* bodies,PrismGearGovernorDeviceV2* g) {
    if(!bodies||!g)return static_cast<int>(cudaErrorInvalidValue);
    for(uint32_t gear=0;gear<4;++gear){
        if(!bodies[gear])return static_cast<int>(cudaErrorInvalidValue);
        cudaGraphNode_t node=nullptr;void* args[]={&g,&gear};
        auto rc=add_kernel_node(&node,bodies[gear],nullptr,0,
            gear==3?reinterpret_cast<void*>(prism_gearbox_trap_kernel):reinterpret_cast<void*>(prism_gearbox_guard_observe_kernel),
            dim3(1,1,1),dim3(1,1,1),gear==3?nullptr:args,0);
        if(rc!=cudaSuccess)return static_cast<int>(rc);
    }
    return 0;
}

#else  // CUDART_VERSION < 12060

extern "C" int prism_gearbox_populate_guard_bodies_v1(cudaGraph_t*,PrismGearGovernorDeviceV2*) {
    return static_cast<int>(cudaErrorNotSupported);
}

extern "C"
int prism_gearbox_populate_switch_bodies_ffi(
    cudaGraph_t*                          /*body_subgraphs*/,
    const InterferometricAdjudicatorFfi*  /*adj*/,
    float*                                /*d_velocities*/,
    uint32_t                              /*n_floats*/,
    const ChronometricStateTensor*        /*cruise*/,
    const float*                          /*d_current_temp*/,
    const float*                          /*d_dt*/,
    float                                 /*target_temp_K*/,
    float                                 /*tau_ps*/)
{
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12060

// ════════════════════════════════════════════════════════════════════
// WP C — Physical Stability Governor: host surface
// ════════════════════════════════════════════════════════════════════
//
// Arming, binding and salience publication are all host-resident writes into
// module globals.  Like `prism_gearbox_init_table_async` they MUST be issued
// BEFORE the capture window opens; each one refuses on a capturing stream
// rather than silently corrupting the graph.

namespace {

/// Fail-closed capture check shared by the governor's host entry points.
/// Returns true when the stream is (or may be) capturing.  A non-success
/// query is treated as capturing: the legacy default stream reports
/// cudaErrorStreamCaptureImplicit while another stream in the same context is
/// capturing, and guessing "not capturing" there would corrupt a graph.
__host__ bool prism_gear_governor_stream_capturing(cudaStream_t s) {
    if (s == nullptr) return false;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    const cudaError_t crc = cudaStreamIsCapturing(s, &cs);
    return (crc != cudaSuccess) || (cs != cudaStreamCaptureStatusNone);
}

}  // anonymous namespace

extern "C"
int prism_gearbox_governor_default_policy(PrismGearGovernorPolicy* out)
{
    if (out == nullptr) return static_cast<int>(cudaErrorInvalidValue);

    memset(out, 0, sizeof(*out));
    out->schema_version                       = PRISM_GEAR_GOVERNOR_SCHEMA_VERSION;
    out->armed                                = 0u;
    out->unknown_topology_cap_gear            = PRISM_GEAR_DEFAULT_UNKNOWN_TOPOLOGY_CAP;
    out->max_upshift_step                     = PRISM_GEAR_DEFAULT_MAX_UPSHIFT_STEP;
    out->min_upshift_dwell_steps              = PRISM_GEAR_DEFAULT_MIN_UPSHIFT_DWELL_STEPS;
    out->instability_lockout_steps            = PRISM_GEAR_DEFAULT_INSTABILITY_LOCKOUT_STEPS;
    out->require_force_guard_for_upshift      = PRISM_GEAR_DEFAULT_REQUIRE_FORCE_GUARD;
    out->require_constraint_guard_for_upshift = PRISM_GEAR_DEFAULT_REQUIRE_CONSTRAINT_GUARD;
    out->salience_max_age_frames              = PRISM_GEAR_DEFAULT_SALIENCE_MAX_AGE_FRAMES;
    out->chunk_span_steps                     = 1u;
    out->hmr_permission                       = PRISM_GEAR_PERMISSION_UNKNOWN;
    out->constraint_permission                = PRISM_GEAR_PERMISSION_UNKNOWN;
    out->gear_dt_fs[0]                        = PRISM_GEAR_DEFAULT_DT_FS_G0;
    out->gear_dt_fs[1]                        = PRISM_GEAR_DEFAULT_DT_FS_G1;
    out->gear_dt_fs[2]                        = PRISM_GEAR_DEFAULT_DT_FS_G2;
    out->gear_dt_fs[3]                        = PRISM_GEAR_DEFAULT_DT_FS_G3;
    out->max_uncorrected_window_fs            = PRISM_GEAR_DEFAULT_MAX_UNCORRECTED_WINDOW_FS;
    out->drift_abort                          = PRISM_GEAR_DEFAULT_DRIFT_ABORT;
    out->drift_upshift_fraction               = PRISM_GEAR_DEFAULT_DRIFT_UPSHIFT_FRACTION;
    out->drift_hold_fraction                  = PRISM_GEAR_DEFAULT_DRIFT_HOLD_FRACTION;
    out->drift_downshift_fraction             = PRISM_GEAR_DEFAULT_DRIFT_DOWNSHIFT_FRACTION;
    out->force_clamp_reference                = PRISM_GEAR_DEFAULT_FORCE_CLAMP_REFERENCE;
    out->force_upshift_fraction               = PRISM_GEAR_DEFAULT_FORCE_UPSHIFT_FRACTION;
    out->force_hold_fraction                  = PRISM_GEAR_DEFAULT_FORCE_HOLD_FRACTION;
    out->force_downshift_fraction             = PRISM_GEAR_DEFAULT_FORCE_DOWNSHIFT_FRACTION;
    out->force_abort_fraction                 = PRISM_GEAR_DEFAULT_FORCE_ABORT_FRACTION;
    out->constraint_failure_hold              = PRISM_GEAR_DEFAULT_CONSTRAINT_FAILURE_HOLD;
    out->constraint_failure_downshift         = PRISM_GEAR_DEFAULT_CONSTRAINT_FAILURE_DOWNSHIFT;
    out->policy_digest                        = 0ull;
    out->cruise_threshold_steps               = PRISM_GEAR_DEFAULT_CRUISE_THRESHOLD_STEPS;
    return static_cast<int>(cudaSuccess);
}

extern "C"
int prism_gearbox_governor_decide_host(
    const PrismGearGovernorPolicy*   policy,
    const PrismGearGuardObservation* observation,
    const PrismTshSalienceRequest*   tsh,
    PrismGearDecision*               out)
{
    if (policy == nullptr || observation == nullptr || out == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    // Same function the SFA kernel calls — a host test therefore exercises the
    // production decision path rather than a parallel re-implementation.
    prism_gearbox_governor_decide(policy, observation, tsh, out);
    return static_cast<int>(cudaSuccess);
}

extern "C" int prism_gearbox_governor_arm_async(const PrismGearGovernorPolicy*, void*) { return static_cast<int>(cudaErrorNotSupported); }
extern "C" int prism_gearbox_governor_bind_measurements_async(const PrismGearGovernorBindings*, void*) { return static_cast<int>(cudaErrorNotSupported); }
extern "C" int prism_gearbox_governor_publish_salience_async(const PrismTshSalienceRequest*, void*) { return static_cast<int>(cudaErrorNotSupported); }
extern "C" int prism_gearbox_governor_snapshot(void*, PrismGearGovernorState*) { return static_cast<int>(cudaErrorNotSupported); }

// One contiguous owned allocation: context, force record, constraint record,
// physical epoch, padding, then three phase-by-H-cluster success epoch arrays.
static constexpr size_t PRISM_GOV_ROWS_OFFSET = 624;
__global__ void prism_governor_init_v2(PrismGearGovernorDeviceV2* g,
    PrismGearGovernorPolicy policy, uint32_t source, uint64_t generation, uint32_t clusters) {
    if (threadIdx.x || blockIdx.x) return;
    uint8_t* base = reinterpret_cast<uint8_t*>(g);
    *g = PrismGearGovernorDeviceV2{};
    g->abi_version = 2; g->byte_size = sizeof(*g); g->source_stream_id = source;
    g->lifecycle = 1; g->allocation_generation = generation; g->binding_epoch = 1;
    g->policy = policy; g->reset_reason = 1;
    g->bindings.force_window = reinterpret_cast<PrismForceWindowV1*>(base + 384);
    g->bindings.constraint_window = reinterpret_cast<PrismConstraintWindowV1*>(base + 448);
    g->bindings.physical_epoch = reinterpret_cast<uint64_t*>(base + 576);
    auto* fw = g->bindings.force_window; *fw = PrismForceWindowV1{};
    fw->schema_version = 1; fw->source_stream_id = source; fw->allocation_generation = generation;
    fw->force_norm = CUDART_NAN_F;
    auto* cw = g->bindings.constraint_window; *cw = PrismConstraintWindowV1{};
    cw->schema_version = 1; cw->source_stream_id = source; cw->allocation_generation = generation;
    cw->n_constraint_clusters = clusters;
    cw->phase_cluster_epochs = reinterpret_cast<uint64_t*>(base + PRISM_GOV_ROWS_OFFSET);
    *g->bindings.physical_epoch = 0;
    auto* cruise = reinterpret_cast<ChronometricStateTensor*>(base + 592);
    *cruise = ChronometricStateTensor{}; cruise->current_gear = 1u; cruise->previous_gear = 1u;
    auto* state = &g->state;
    state->schema_version = PRISM_GEAR_GOVERNOR_SCHEMA_VERSION;
    state->authorized_gear = state->ceiling_gear = state->requested_gear = PRISM_GEAR_UNSET;
    state->observed_force_norm = state->observed_drift = CUDART_NAN_F;
    state->observed_constraint_failures = PRISM_GEAR_CONSTRAINT_UNMEASURED;
    state->governor_armed = policy.armed; state->policy_schema_version = policy.schema_version;
    state->policy_digest = policy.policy_digest;
    state->salience.schema_version = PRISM_TSH_SALIENCE_SCHEMA_VERSION;
    state->salience.state = PRISM_TSH_SALIENCE_UNSET;
    state->salience.requested_gear = PRISM_GEAR_UNSET;
}
extern "C" int prism_gearbox_governor_create_v2(PrismGearGovernorDeviceV2** out,
    const PrismGearGovernorPolicy* policy, uint32_t source, uint64_t generation,
    uint32_t clusters, void* stream) {
    if (!out || !policy || !stream || !generation ||
        policy->schema_version != PRISM_GEAR_GOVERNOR_SCHEMA_VERSION || policy->chunk_span_steps == 0)
        return static_cast<int>(cudaErrorInvalidValue);
    *out = nullptr;
    auto s = static_cast<cudaStream_t>(stream);
    if (prism_gear_governor_stream_capturing(s)) return static_cast<int>(cudaErrorStreamCaptureUnsupported);
    const size_t bytes = PRISM_GOV_ROWS_OFFSET + static_cast<size_t>(clusters) * 3 * sizeof(uint64_t);
    auto rc = cudaMalloc(reinterpret_cast<void**>(out), bytes);
    if (rc != cudaSuccess) return static_cast<int>(rc);
    rc = cudaMemsetAsync(*out, 0, bytes, s);
    if (rc == cudaSuccess) {
        // Kernel parameters are copied by value before the call returns. There
        // is no asynchronous HtoD copy from a temporary host policy/seed.
        prism_governor_init_v2<<<1, 1, 0, s>>>(*out, *policy, source, generation, clusters);
        rc = cudaGetLastError();
    }
    if (rc != cudaSuccess) {
        // Allocation deliberately remains owned by the caller on failure; it
        // may be released only after a successful drain of the init stream.
        return static_cast<int>(rc);
    }
    return 0;
}
__global__ void prism_constraint_window_begin_v1(PrismGearGovernorDeviceV2* g) {
    if (threadIdx.x || blockIdx.x || !g) return;
    auto* w = g->bindings.constraint_window;
    auto* f = g->bindings.force_window;
    const uint64_t epoch = ++*g->bindings.physical_epoch;
    if (!epoch) { asm volatile("trap;"); return; }
    w->status = PRISM_WINDOW_OPEN; w->window_epoch = epoch;
    w->first_step_epoch = w->last_step_epoch = epoch; w->sealed_epoch = 0;
    w->expected_physical_steps = 1; w->completed_physical_steps = 0;
    w->fatal_failure_count = 0; w->phase_mask = 0;
    for (int p = 0; p < 3; ++p) { w->expected_cluster_solves[p] = w->n_constraint_clusters; w->completed_cluster_solves[p] = 0; }
    f->status = PRISM_WINDOW_OPEN; f->window_epoch = epoch;
    f->first_step_epoch = f->last_step_epoch = epoch; f->force_norm = CUDART_NAN_F;
}
__global__ void prism_constraint_window_seal_v1(PrismGearGovernorDeviceV2* g) {
    if (!g) return;
    auto* w = g->bindings.constraint_window;
    // Exactly one block. All cluster rows are observed after the final solver
    // and work reduction on the same stream, without replacing solver health.
    __shared__ unsigned long long coverage[3][256];
    const unsigned int lane = threadIdx.x;
    for (int p = 0; p < 3; ++p) {
        unsigned long long count = 0;
        for (uint32_t c = lane; c < w->n_constraint_clusters; c += blockDim.x)
            count += w->phase_cluster_epochs[static_cast<uint64_t>(p)*w->n_constraint_clusters+c] == w->window_epoch;
        coverage[p][lane] = count;
    }
    __syncthreads();
    for (unsigned int d = 128; d; d >>= 1) {
        if (lane < d) for (int p = 0; p < 3; ++p) coverage[p][lane] += coverage[p][lane+d];
        __syncthreads();
    }
    if (!lane) {
        for (int p = 0; p < 3; ++p) {
            w->completed_cluster_solves[p] = coverage[p][0];
            if (coverage[p][0] == w->n_constraint_clusters) w->phase_mask |= 1u << p;
        }
        w->completed_physical_steps = 1; w->sealed_epoch = *g->bindings.physical_epoch;
        w->status = !w->n_constraint_clusters ? PRISM_WINDOW_NOT_APPLICABLE :
            (w->fatal_failure_count ? PRISM_WINDOW_FAILED_DIAGNOSTIC :
             (w->phase_mask == 7 ? PRISM_WINDOW_COMPLETE : PRISM_WINDOW_INCOMPLETE_COVERAGE));
    }
}
extern "C" int prism_gearbox_governor_begin_step_v2(PrismGearGovernorDeviceV2* g, void* stream) {
    if (!g || !stream) return static_cast<int>(cudaErrorInvalidValue);
    prism_constraint_window_begin_v1<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(g);
    return static_cast<int>(cudaGetLastError());
}
extern "C" int prism_gearbox_governor_seal_step_v2(PrismGearGovernorDeviceV2* g, void* stream) {
    if (!g || !stream) return static_cast<int>(cudaErrorInvalidValue);
    prism_constraint_window_seal_v1<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(g);
    return static_cast<int>(cudaGetLastError());
}
__global__ void prism_force_window_complete_v1(PrismGearGovernorDeviceV2* g, uint32_t producer) {
    if (threadIdx.x || blockIdx.x) return;
    auto* f = g->bindings.force_window;
    f->window_epoch = f->first_step_epoch = f->last_step_epoch = *g->bindings.physical_epoch;
    f->producer_phase = producer; f->status = PRISM_WINDOW_COMPLETE;
}
extern "C" int prism_gearbox_governor_force_complete_v2(PrismGearGovernorDeviceV2* g, uint32_t producer, void* stream) {
    if (!g || !stream) return static_cast<int>(cudaErrorInvalidValue);
    prism_force_window_complete_v1<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(g, producer);
    return static_cast<int>(cudaGetLastError());
}
// Same physical L2 norm as ZSTR: sqrt(sum_i(Fx^2+Fy^2+Fz^2)), kcal/mol/A.
// Used when ZSTR is disabled; this is an actual native force reduction.
__global__ void prism_force_window_reduce_v1(PrismGearGovernorDeviceV2* g, const float3* forces, uint32_t n) {
    __shared__ float sums[256];
    const uint32_t t = threadIdx.x;
    float sum = 0;
    for (uint32_t i = t; i < n; i += 256) { const float3 f = forces[i]; sum += f.x*f.x+f.y*f.y+f.z*f.z; }
    sums[t] = sum; __syncthreads();
    for (uint32_t d = 128; d; d >>= 1) { if (t < d) sums[t] += sums[t+d]; __syncthreads(); }
    if (!t) g->bindings.force_window->force_norm = sqrtf(sums[0]);
}
extern "C" int prism_gearbox_governor_measure_force_v2(PrismGearGovernorDeviceV2* g,
    const void* forces, uint32_t n, void* stream) {
    if (!g || !forces || !n || !stream) return static_cast<int>(cudaErrorInvalidValue);
    auto s = static_cast<cudaStream_t>(stream);
    prism_force_window_reduce_v1<<<1, 256, 0, s>>>(g, static_cast<const float3*>(forces), n);
    auto rc = cudaGetLastError(); if (rc != cudaSuccess) return static_cast<int>(rc);
    return prism_gearbox_governor_force_complete_v2(g, 1, stream);
}
extern "C" int prism_gearbox_governor_snapshot_v2(const PrismGearGovernorDeviceV2* g,
    void* stream, uint32_t timeout_ms, PrismGearGovernorSnapshotV2* out, PrismConstraintWindowV1* constraints) {
    if (!g || !stream || !out || !constraints || !timeout_ms) return static_cast<int>(cudaErrorInvalidValue);
    auto s = static_cast<cudaStream_t>(stream);
    if (prism_gear_governor_stream_capturing(s)) return static_cast<int>(cudaErrorStreamCaptureUnsupported);
    uint64_t waited = 0;
    int rc = prism_gearbox_bounded_drain(s, static_cast<uint64_t>(timeout_ms)*1000, &waited);
    if (rc) return rc;
    PrismGearGovernorDeviceV2 host{};
    auto cr = cudaMemcpy(&host, g, sizeof(host), cudaMemcpyDeviceToHost);
    if (cr != cudaSuccess) return static_cast<int>(cr);
    if (host.abi_version != 2 || host.byte_size != sizeof(host)) return static_cast<int>(cudaErrorInvalidValue);
    memcpy(out, &host, 32);
    out->byte_size = sizeof(*out);
    memcpy(&out->state, &host.state, sizeof(host)-offsetof(PrismGearGovernorDeviceV2, state));
    return static_cast<int>(cudaMemcpy(constraints, host.bindings.constraint_window, sizeof(*constraints), cudaMemcpyDeviceToHost));
}
extern "C" int prism_gearbox_governor_destroy_v2(PrismGearGovernorDeviceV2* g, void* stream, uint32_t timeout_ms) {
    if (!g || !stream || !timeout_ms) return static_cast<int>(cudaErrorInvalidValue);
    auto s = static_cast<cudaStream_t>(stream);
    if (prism_gear_governor_stream_capturing(s)) return static_cast<int>(cudaErrorStreamCaptureUnsupported);
    uint64_t waited = 0;
    int rc = prism_gearbox_bounded_drain(s, static_cast<uint64_t>(timeout_ms)*1000, &waited);
    if (rc) return rc; // Quarantine: never free storage whose completion is unknown.
    return static_cast<int>(cudaFree(g));
}

extern "C" int prism_gearbox_stream_drain_v2(void* stream, uint32_t timeout_ms) {
    if (!stream || !timeout_ms) return static_cast<int>(cudaErrorInvalidValue);
    auto s = static_cast<cudaStream_t>(stream);
    if (prism_gear_governor_stream_capturing(s)) return static_cast<int>(cudaErrorStreamCaptureUnsupported);
    uint64_t waited = 0;
    return prism_gearbox_bounded_drain(s, static_cast<uint64_t>(timeout_ms) * 1000, &waited);
}

// Observation work covers every completed physical step since the preceding
// SFA consumption. Offset 584 is the reserved aligned cell between the epoch
// counter and cruise state; it never aliases any atom work row.
__device__ bool prism_guard_owner_valid(PrismPhysicalGuardV1* guard);
__global__ void prism_governor_accumulate_work_v2(PrismGearGovernorDeviceV2* g,const double* step_work) {
    if(threadIdx.x||blockIdx.x)return;
    double interval_work = *step_work;
    if (g->bindings.reserved) {
        auto* guard = reinterpret_cast<PrismPhysicalGuardV1*>(g->bindings.reserved);
        if (!prism_guard_owner_valid(guard) || guard->bindings.governor != g ||
            guard->bindings.step_work != step_work) {
            guard->snapshot.status = PRISM_GUARD_OWNER_MISMATCH;
            asm volatile("trap;"); return;
        }
        // nhs_reduce_external_work_components atomicAdds into the cumulative
        // replay window. Add only this physical body's increment to the SFA
        // window; pending transition work is added separately by evaluation.
        interval_work -= guard->work_at_begin;
    }
    auto* accumulated=reinterpret_cast<double*>(reinterpret_cast<uint8_t*>(g)+584);
    *accumulated+=interval_work;
}
extern "C" int prism_gearbox_governor_accumulate_work_v2(PrismGearGovernorDeviceV2* g,const double* step_work,void* stream) {
    if(!g||!step_work||!stream)return (int)cudaErrorInvalidValue;
    prism_governor_accumulate_work_v2<<<1,1,0,(cudaStream_t)stream>>>(g,step_work);return (int)cudaGetLastError();
}

__device__ bool prism_guard_owner_valid(PrismPhysicalGuardV1* guard) {
    if (!guard) return false;
    const auto& b = guard->bindings;
    return b.abi == 1u && b.byte_size == sizeof(b) && b.governor && b.n_atoms &&
        b.source == b.governor->source_stream_id && b.generation == b.governor->allocation_generation &&
        b.governor->bindings.reserved == reinterpret_cast<uint64_t>(guard) &&
        b.forces && b.potential && b.velocities && b.masses && b.dt && b.step_work;
}

__device__ void prism_guard_decide(PrismPhysicalGuardV1* guard, bool measured) {
    auto& s = guard->snapshot;
    auto* g = guard->bindings.governor;
    if (s.dt_status==PRISM_GUARD_DT_SHARED_SELECTED && g->policy.armed) {
        s.status=PRISM_GUARD_DT_CONFLICT; asm volatile("trap;"); return;
    }
    auto* cruise = reinterpret_cast<ChronometricStateTensor*>(reinterpret_cast<uint8_t*>(g) + 592);
    PrismGearGuardObservation obs{};
    const uint32_t request = prism_guard_requested_gear(s);
    obs.adjudication_code = request == PRISM_GEAR_ABORT ? 2u : s.science_code;
    const bool established = s.physical_eval_epoch != 0;
    const uint32_t history_status=prism_guard_energy_history_status(established,guard->previous_kinetic);
    // Invalid established reference is a physical measurement failure even when
    // the gear policy is disarmed. It must never masquerade as first-step bootstrap.
    obs.nonfinite_energy = measured && (!isfinite(s.current_pe) ||
        !isfinite(guard->current_hamiltonian) || !isfinite(s.observed_work) ||
        !isfinite(guard->current_kinetic) || guard->current_kinetic<0.0 ||
        (established && (history_status!=PRISM_ENERGY_ESTABLISHED_VALID || !isfinite(s.observed_drift))));
    obs.drift_measured = measured && established;
    if(measured && obs.nonfinite_energy) {
        printf("PRISM_ENERGY_REFERENCE_INVALID source=%u generation=%llu epoch=%llu contract=%u history_status=%u "
               "u_previous=%.17g u_current=%.17g k_previous=%.17g k_current=%.17g work=%.17g component_residual=%.17g normalized=%.17g\n",
               guard->bindings.source,(unsigned long long)guard->bindings.generation,
               (unsigned long long)*g->bindings.physical_epoch,PRISM_ENERGY_NORMALIZATION_CONTRACT,history_status,
               s.previous_pe,s.current_pe,guard->previous_kinetic,guard->current_kinetic,s.observed_work,
               prism_guard_component_residual(s.previous_pe,s.current_pe,guard->previous_kinetic,guard->current_kinetic,s.observed_work),s.observed_drift);
    }
    obs.drift = obs.drift_measured ? static_cast<float>(s.observed_drift) : CUDART_NAN_F;
    obs.force_measured = measured && g->force_status == PRISM_WINDOW_COMPLETE;
    obs.force_norm = obs.force_measured ? g->bindings.force_window->force_norm : CUDART_NAN_F;
    obs.constraint_measured = measured &&
        (g->constraint_status == PRISM_WINDOW_COMPLETE || g->constraint_status == PRISM_WINDOW_FAILED_DIAGNOSTIC);
    obs.constraint_failures = obs.constraint_measured ? g->bindings.constraint_window->fatal_failure_count : PRISM_GEAR_CONSTRAINT_UNMEASURED;
    obs.current_gear = s.applied_gear; obs.requested_gear = request;
    obs.stable_dwell_steps = g->state.stable_dwell_steps;
    obs.steps_since_instability = g->state.steps_since_instability;
    obs.frame = static_cast<uint32_t>(*g->bindings.physical_epoch);
    if (*g->bindings.physical_epoch > UINT32_MAX) obs.nonfinite_energy = 1;
    // --gearbox off selects fixed-step execution. Keep the physical observer
    // alive, but do not apply science requests, governor thresholds, or gear
    // transitions from a policy the operator explicitly left disarmed.
    if (!g->policy.armed) {
        const bool invalid_force = obs.force_measured && !isfinite(obs.force_norm);
        const bool failed_constraints = obs.constraint_measured && obs.constraint_failures;
        if (obs.nonfinite_energy || invalid_force || failed_constraints) {
            s.status = PRISM_GUARD_NONFINITE;
            printf("PRISM_INVALID_PHYSICAL_STATE epoch=%llu energy_nonfinite=%u "
                   "force_nonfinite=%u constraint_failures=%u\n",
                   static_cast<unsigned long long>(*g->bindings.physical_epoch),
                   obs.nonfinite_energy, static_cast<unsigned>(invalid_force),
                   obs.constraint_failures);
            asm volatile("trap;"); return;
        }
        auto& gs = g->state;
        gs.authorized_gear = gs.ceiling_gear = s.applied_gear;
        gs.requested_gear = s.applied_gear;
        gs.binding_gate = PRISM_GEAR_GATE_NONE;
        gs.gate_verdicts = gs.emergency_flags = 0;
        if (measured) {
            gs.observed_drift = obs.drift;
            gs.observed_force_norm = obs.force_norm;
            gs.observed_constraint_failures = obs.constraint_failures;
            if (*g->bindings.physical_epoch <= 2u) {
                printf("PRISM_FIXED_STEP epoch=%llu dt=%.9g hamiltonian=%.17g "
                       "work=%.17g drift=%.9g force=%.9g\n",
                       static_cast<unsigned long long>(*g->bindings.physical_epoch),
                       static_cast<double>(*guard->bindings.dt), guard->current_hamiltonian,
                       s.observed_work, s.observed_drift, static_cast<double>(obs.force_norm));
            }
        }
        guard->transition_from = guard->transition_to = s.applied_gear;
        s.transition_pending = 0;
        return;
    }
    PrismGearDecision d{};
    prism_gearbox_governor_decide(&g->policy, &obs, &g->state.salience, &d);
    auto& gs = g->state;
    ++gs.decision_epoch; ++g->decision_epoch;
    gs.authorized_gear = d.authorized_gear; gs.ceiling_gear = d.ceiling_gear;
    gs.requested_gear = request; gs.binding_gate = d.binding_gate;
    gs.gate_verdicts = d.gate_verdicts; gs.downshift_latency_steps = d.downshift_latency_steps;
    // A pre-kick admission does not earn another physical step of dwell.
    if (measured) {
        gs.stable_dwell_steps = d.next_stable_dwell_steps;
        gs.steps_since_instability = d.next_steps_since_instability;
        gs.observed_drift = obs.drift;
        gs.observed_force_norm = obs.force_norm;
        gs.observed_constraint_failures = obs.constraint_failures;
    }
    gs.emergency_flags = d.emergency_flags;
    if (d.authorized_gear > PRISM_GEAR_MAX_ACTIVE) {
        s.status = PRISM_GUARD_EMERGENCY; cruise->current_gear = PRISM_GEAR_ABORT;
        printf("PRISM_DEVICE_EMERGENCY source=%u generation=%llu native_step_bound=%u native_step=%llu measured=%u epoch=%llu flags=%u force=%.9g drift=%.9g "
               "previous_h=%.17g current_h=%.17g pe=%.17g ke=%.17g work=%.17g "
               "cumulative_work=%.17g work_at_begin=%.17g body_work=%.17g carried_transition_work=%.17g expected_h=%.17g signed_residual=%.17g "
               "normalization_contract=%u history_status=%u previous_pe=%.17g previous_ke=%.17g current_ke=%.17g "
               "component_signed_residual=%.17g kinetic_reference=%.17g legacy_h_ratio=%.17g "
               "fatal_constraints=%u gear=%u dt=%.9g\n",
               guard->bindings.source, static_cast<unsigned long long>(guard->bindings.generation),
               static_cast<unsigned>(guard->time_step != nullptr),
               guard->time_step ? static_cast<unsigned long long>(*guard->time_step) : 0ull,
               static_cast<unsigned>(measured),
               static_cast<unsigned long long>(*g->bindings.physical_epoch),
               d.emergency_flags, static_cast<double>(obs.force_norm),
               s.observed_drift, guard->previous_hamiltonian, guard->current_hamiltonian,
               s.current_pe, guard->current_hamiltonian - s.current_pe, s.observed_work,
               *guard->bindings.step_work, guard->work_at_begin,
               *guard->bindings.step_work - guard->work_at_begin,
               s.observed_work - (*guard->bindings.step_work - guard->work_at_begin),
               guard->previous_hamiltonian + s.observed_work,
               guard->current_hamiltonian - (guard->previous_hamiltonian + s.observed_work),
               PRISM_ENERGY_NORMALIZATION_CONTRACT,history_status,s.previous_pe,guard->previous_kinetic,guard->current_kinetic,
               prism_guard_component_residual(s.previous_pe,s.current_pe,guard->previous_kinetic,guard->current_kinetic,s.observed_work),
               guard->previous_kinetic,prism_guard_relative_residual(guard->previous_hamiltonian,guard->current_hamiltonian,s.observed_work),
               obs.constraint_failures, s.applied_gear, static_cast<double>(*guard->bindings.dt));
        assert(false && "PRISM_DEVICE_EMERGENCY: physical guard rejected observation");
        asm volatile("trap;"); return;
    }
    guard->transition_from = s.applied_gear; guard->transition_to = d.authorized_gear;
    s.transition_pending = guard->transition_from != guard->transition_to;
    if (d.authorized_gear < s.applied_gear) gs.last_downshift_epoch = gs.decision_epoch;
    if (d.authorized_gear > s.applied_gear) gs.last_upshift_epoch = gs.decision_epoch;
}

__global__ void prism_physical_guard_init_v1(PrismPhysicalGuardV1* guard,
    PrismPhysicalGuardBindingsV1 b) {
    if (threadIdx.x || blockIdx.x) return;
    *guard = PrismPhysicalGuardV1{};
    guard->bindings = b;
    guard->transition_work_rows = reinterpret_cast<double*>(reinterpret_cast<uint8_t*>(guard) + sizeof(*guard));
    auto& s = guard->snapshot;
    s.abi = 1; s.byte_size = sizeof(s); s.source = b.source; s.generation = b.generation;
    s.science_code = s.science_request = s.science_override = s.host_ceiling = PRISM_GEAR_UNSET;
    s.initial_gear = s.applied_gear = PRISM_GEAR_UNSET;
    s.previous_pe = s.current_pe = s.observed_work = s.observed_drift = CUDART_NAN;
    guard->previous_hamiltonian = guard->current_hamiltonian = CUDART_NAN;
    guard->work_at_begin = CUDART_NAN;
    guard->previous_kinetic = guard->current_kinetic = CUDART_NAN;
    b.governor->bindings.reserved = reinterpret_cast<uint64_t>(guard);
}

extern "C" int prism_physical_guard_create_v1(uint64_t* out,
    const PrismPhysicalGuardBindingsV1* bindings, void* stream) {
    if (!out || !bindings || !stream || bindings->abi != 1 || bindings->byte_size != sizeof(*bindings) ||
        !bindings->n_atoms || !bindings->governor || !bindings->forces || !bindings->potential ||
        !bindings->velocities || !bindings->masses || !bindings->dt || !bindings->step_work)
        return static_cast<int>(cudaErrorInvalidValue);
    *out = 0;
    auto s = static_cast<cudaStream_t>(stream);
    if (prism_gear_governor_stream_capturing(s)) return static_cast<int>(cudaErrorStreamCaptureUnsupported);
    PrismPhysicalGuardV1* guard = nullptr;
    auto rc = cudaMalloc(reinterpret_cast<void**>(&guard), sizeof(*guard) + static_cast<size_t>(bindings->n_atoms)*sizeof(double));
    if (rc != cudaSuccess) return static_cast<int>(rc);
    *out = reinterpret_cast<uint64_t>(guard); // caller retains even if launch fails
    prism_physical_guard_init_v1<<<1,1,0,s>>>(guard, *bindings);
    return static_cast<int>(cudaGetLastError());
}


// Additive common-selected measurement admission. No discrete gear selection,
// velocity rescaling, history reset, or acceptance-threshold change occurs.
__global__ void prism_physical_guard_begin_selected_v1(PrismPhysicalGuardV1* guard,
    const BiprismSharedStepFfi* shared, uint64_t expected_generation,
    uint32_t leaf_id, uint32_t expected_source) {
    if (threadIdx.x || blockIdx.x) return;
    auto& s=guard->snapshot;
    if (!prism_guard_owner_valid(guard) || !shared) {
        s.status=PRISM_GUARD_OWNER_MISMATCH; asm volatile("trap;"); return;
    }
    const uint32_t admission=prism_guard_selected_step_status_v1(
        *guard,*shared,expected_generation,leaf_id,expected_source);
    if (admission!=PRISM_GUARD_READY) {
        s.status=admission; asm volatile("trap;"); return;
    }
    guard->last_begin_epoch=*guard->bindings.governor->bindings.physical_epoch;
    guard->initializing=1u;
    guard->work_at_begin=*guard->bindings.step_work;
    if (s.status==PRISM_GUARD_UNINITIALIZED) s.initial_gear=s.applied_gear=0u;
    s.dt_status=PRISM_GUARD_DT_SHARED_SELECTED;
    s.status=PRISM_GUARD_READY;
    prism_guard_decide(guard,false);
}

__global__ void prism_physical_guard_begin_v1(PrismPhysicalGuardV1* guard) {
    if (threadIdx.x || blockIdx.x) return;
    auto& s = guard->snapshot;
    if (!prism_guard_owner_valid(guard)) { s.status = PRISM_GUARD_OWNER_MISMATCH; asm volatile("trap;"); return; }
    // A captured selected profile cannot silently switch back to ordinary admission.
    if (s.dt_status==PRISM_GUARD_DT_SHARED_SELECTED) {
        s.status=PRISM_GUARD_DT_CONFLICT; asm volatile("trap;"); return;
    }
    auto* g = guard->bindings.governor;
    const uint64_t epoch = *g->bindings.physical_epoch;
    if (!prism_guard_next_epoch(epoch,s.physical_eval_epoch,guard->last_begin_epoch)) {
        s.status = PRISM_GUARD_EPOCH_MISMATCH; asm volatile("trap;"); return;
    }
    // Capacity is admitted BEFORE positions advance; a full journal cannot
    // silently discard an executed physical step.
    if (guard->time_rows && guard->time_count >= guard->time_capacity) {
        s.status=PRISM_GUARD_EPOCH_MISMATCH;asm volatile("trap;");return;
    }
    guard->last_begin_epoch = epoch; guard->initializing = 1;
    // bindings.step_work is fused_engine.d_external_work_window, whose
    // producer nhs_reduce_external_work_components accumulates via atomicAdd
    // across a whole replay. Freeze its value before this physical body.
    guard->work_at_begin = *guard->bindings.step_work;
    const float dt = *guard->bindings.dt;
    if (s.status == PRISM_GUARD_UNINITIALIZED) {
        s.initial_gear = s.applied_gear = prism_guard_gear_for_dt(g->policy, dt);
        if (s.applied_gear > PRISM_GEAR_MAX_ACTIVE || g->policy.chunk_span_steps != 1u) {
            s.status = PRISM_GUARD_DT_CONFLICT; asm volatile("trap;"); return;
        }
    } else if (s.status != PRISM_GUARD_COMPLETE || static_cast<double>(dt) != s.admitted_dt) {
        s.status = PRISM_GUARD_DT_CONFLICT; asm volatile("trap;"); return;
    }
    s.status = PRISM_GUARD_READY;
    prism_guard_decide(guard, false);
}

__global__ void prism_physical_guard_evaluate_v1(PrismPhysicalGuardV1* guard) {
    __shared__ double pe[256];
    __shared__ double ke[256];
    const auto& b = guard->bindings;
    const uint32_t lane = threadIdx.x;
    double potential = 0, kinetic = 0;
    for (uint32_t i = lane; i < b.n_atoms; i += blockDim.x) {
        potential += b.potential[i];
        const double vx = static_cast<double>(b.velocities[3u*i]);
        const double vy = static_cast<double>(b.velocities[3u*i+1u]);
        const double vz = static_cast<double>(b.velocities[3u*i+2u]);
        // Velocities are A/ps and masses are amu; 418.4 converts the
        // kinetic energy to kcal/mol, matching PE and step-work units.
        kinetic += 0.5 * static_cast<double>(b.masses[i]) *
            (vx*vx + vy*vy + vz*vz) / 418.4;
    }
    pe[lane] = potential; ke[lane] = kinetic; __syncthreads();
    // Fixed lane/stride reduction order for both energy components.
    for (uint32_t d=128; d; d>>=1) {
        if(lane<d) { pe[lane]+=pe[lane+d]; ke[lane]+=ke[lane+d]; }
        __syncthreads();
    }
    if (lane) return;
    auto& s = guard->snapshot;
    if (!prism_guard_owner_valid(guard)) { s.status = PRISM_GUARD_OWNER_MISMATCH; asm volatile("trap;"); return; }
    auto* g = b.governor;
    // Reject a changed policy before any physical-time or energy-row commit.
    if (s.dt_status==PRISM_GUARD_DT_SHARED_SELECTED && g->policy.armed) {
        s.status=PRISM_GUARD_DT_CONFLICT; asm volatile("trap;"); return;
    }
    const uint64_t epoch = *g->bindings.physical_epoch;
    if (s.status != PRISM_GUARD_READY || epoch != guard->last_begin_epoch || epoch != s.physical_eval_epoch + 1) {
        s.status = PRISM_GUARD_EPOCH_MISMATCH; asm volatile("trap;"); return;
    }
    g->force_status = prism_force_window_status(g->bindings.force_window,b.source,b.generation,epoch);
    g->constraint_status = prism_constraint_window_status(g->bindings.constraint_window,b.source,b.generation,epoch);
    const auto* cw=g->bindings.constraint_window;
    if(g->force_status!=PRISM_WINDOW_COMPLETE || cw->window_epoch!=epoch || cw->sealed_epoch!=epoch ||
        (g->constraint_status!=PRISM_WINDOW_COMPLETE && g->constraint_status!=PRISM_WINDOW_NOT_APPLICABLE &&
         g->constraint_status!=PRISM_WINDOW_FAILED_DIAGNOSTIC)) {
        s.status=PRISM_GUARD_MEASUREMENT_INCOMPLETE;asm volatile("trap;");return;
    }
    s.force_status = g->force_status; s.constraint_status = g->constraint_status;
    g->observed_force_epoch = g->force_status == PRISM_WINDOW_COMPLETE ? epoch : 0;
    g->observed_constraint_epoch = (g->constraint_status == PRISM_WINDOW_COMPLETE || g->constraint_status == PRISM_WINDOW_FAILED_DIAGNOSTIC) ? epoch : 0;
    g->last_physical_epoch = epoch;
    s.current_pe = pe[0];
    guard->current_kinetic = ke[0];
    const double body_work=*b.step_work-guard->work_at_begin;
    const double carried_work=s.pending_interval_work;
    s.observed_work = body_work + carried_work;
    guard->current_hamiltonian = pe[0] + ke[0];
    *reinterpret_cast<double*>(reinterpret_cast<uint8_t*>(g)+584) += s.pending_interval_work;
    s.pending_interval_work = 0;
    const bool established=s.physical_eval_epoch!=0;
    const uint32_t history_status=prism_guard_energy_history_status(established,guard->previous_kinetic);
    const double component_residual=established ? prism_guard_component_residual(
        s.previous_pe,s.current_pe,guard->previous_kinetic,guard->current_kinetic,s.observed_work) : CUDART_NAN;
    const double legacy_ratio=established ? prism_guard_relative_residual(
        guard->previous_hamiltonian,guard->current_hamiltonian,s.observed_work) : CUDART_NAN;
    s.observed_drift = established ? prism_guard_kinetic_normalized_residual(
        s.previous_pe,s.current_pe,guard->previous_kinetic,guard->current_kinetic,s.observed_work) : CUDART_NAN;
    // All physical phases and constraint seals completed. The dt frozen after
    // pre-kick admission is the value read by every position/velocity phase;
    // record it BEFORE the next guard transition can publish a different dt.
    if (guard->time_rows) {
        const uint64_t step=*guard->time_step;
        if (!guard->energy_rows || !prism_physical_time_commit(guard,step,static_cast<double>(*b.dt))) {
            s.status=PRISM_GUARD_DT_CONFLICT;asm volatile("trap;");return;
        }
        guard->energy_rows[guard->time_count-1u]={1u,sizeof(PrismPhysicalEnergyBalanceRecordV1),b.source,
            history_status,PRISM_ENERGY_NORMALIZATION_CONTRACT,0u,b.generation,step,epoch,s.physical_eval_epoch,
            s.previous_pe,s.current_pe,guard->previous_kinetic,guard->current_kinetic,
            body_work,carried_work,s.observed_work,component_residual,
            guard->previous_kinetic,s.observed_drift,legacy_ratio,
            guard->step_dt_ps,guard->cumulative_time_ps};
    }
    guard->initializing = 0;
    prism_guard_decide(guard, true);
    s.previous_pe = s.current_pe;
    guard->previous_hamiltonian = guard->current_hamiltonian;
    guard->previous_kinetic = guard->current_kinetic;
    s.energy_valid = isfinite(guard->current_hamiltonian);
    const uint64_t gap = epoch - s.physical_eval_epoch;
    if (gap > s.max_evaluation_gap) s.max_evaluation_gap = gap;
    s.physical_eval_epoch = epoch; ++s.physical_decisions;
    s.science_age_physical_steps = epoch - s.science_epoch;
}

__global__ void prism_physical_guard_transition_v1(PrismPhysicalGuardV1* guard) {
    __shared__ double work[256];
    const uint32_t lane = threadIdx.x;
    auto& s = guard->snapshot;
    const auto& b = guard->bindings;
    double sum = 0;
    if (s.status != PRISM_GUARD_READY) return;
    if (s.dt_status==PRISM_GUARD_DT_SHARED_SELECTED &&
        (b.governor->policy.armed || s.transition_pending ||
         guard->transition_from!=0u || guard->transition_to!=0u || s.applied_gear!=0u)) {
        if (!lane) { s.status=PRISM_GUARD_DT_CONFLICT; asm volatile("trap;"); }
        return;
    }
    if (s.transition_pending) {
        const float ratio = d_rescale_ratios[guard->transition_from*3u+guard->transition_to];
        for (uint32_t i=lane; i<b.n_atoms; i+=blockDim.x) {
            float before[3],after[3];
            for(int j=0;j<3;++j) { before[j]=b.velocities[3u*i+j]; after[j]=before[j]*ratio; }
            const double delta = prism_guard_transition_delta_k(b.masses[i],before,after);
            if (!isfinite(delta) || !isfinite(b.masses[i]) || b.masses[i]<=0) {
                s.status=PRISM_GUARD_NONFINITE; asm volatile("trap;"); return;
            }
            for(int j=0;j<3;++j) b.velocities[3u*i+j]=after[j];
            guard->transition_work_rows[i]=delta; sum+=delta;
        }
    }
    work[lane]=sum; __syncthreads();
    for(uint32_t d=128;d;d>>=1){if(lane<d)work[lane]+=work[lane+d];__syncthreads();}
    if(lane)return;
    s.last_transition_work = work[0];
    if(s.transition_pending) {
        if(!isfinite(work[0])) {s.status=PRISM_GUARD_NONFINITE;asm volatile("trap;");return;}
        ++s.transition_count; s.total_transition_work+=work[0]; s.pending_interval_work+=work[0];
    }
    s.applied_gear=guard->transition_to;
    if (b.governor->policy.armed)
        *b.dt=b.governor->policy.gear_dt_fs[s.applied_gear]*0.001f;
    s.admitted_dt=static_cast<double>(*b.dt);
    if (s.dt_status!=PRISM_GUARD_DT_SHARED_SELECTED) s.dt_status=PRISM_GUARD_DT_ORDINARY;
    s.transition_pending=0;
    if (guard->initializing && guard->time_rows) {
        guard->step_dt_ps=static_cast<double>(*b.dt);
        guard->step_at_begin=*guard->time_step;
        if (guard->step_at_begin!=guard->time_count+1u) {
            s.status=PRISM_GUARD_EPOCH_MISMATCH;asm volatile("trap;");return;
        }
    }
    auto* cruise=reinterpret_cast<ChronometricStateTensor*>(reinterpret_cast<uint8_t*>(b.governor)+592);
    cruise->previous_gear=guard->transition_from;cruise->current_gear=s.applied_gear;cruise->v_prev=s.current_pe;
    s.status=guard->initializing?PRISM_GUARD_READY:PRISM_GUARD_COMPLETE;
}

extern "C" int prism_physical_guard_begin_step_v1(uint64_t ptr,void* stream) {
    if(!ptr||!stream)return static_cast<int>(cudaErrorInvalidValue);
    auto* guard=reinterpret_cast<PrismPhysicalGuardV1*>(ptr);auto s=static_cast<cudaStream_t>(stream);
    prism_physical_guard_begin_v1<<<1,1,0,s>>>(guard);
    auto rc=cudaGetLastError();if(rc!=cudaSuccess)return static_cast<int>(rc);
    prism_physical_guard_transition_v1<<<1,256,0,s>>>(guard);return static_cast<int>(cudaGetLastError());
}

extern "C" int prism_physical_guard_begin_selected_step_v1(uint64_t ptr,
    const BiprismSharedStepFfi* shared, uint64_t expected_generation,
    uint32_t leaf_id, uint32_t expected_source, void* stream) {
    if (!ptr || !shared || !expected_generation || leaf_id>=8u || !stream)
        return static_cast<int>(cudaErrorInvalidValue);
    auto* guard=reinterpret_cast<PrismPhysicalGuardV1*>(ptr);
    auto s=static_cast<cudaStream_t>(stream);
    prism_physical_guard_begin_selected_v1<<<1,1,0,s>>>(
        guard,shared,expected_generation,leaf_id,expected_source);
    auto rc=cudaGetLastError(); if(rc!=cudaSuccess)return static_cast<int>(rc);
    prism_physical_guard_transition_v1<<<1,256,0,s>>>(guard);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int prism_physical_guard_evaluate_step_v1(uint64_t ptr,void* stream) {
    if(!ptr||!stream)return static_cast<int>(cudaErrorInvalidValue);
    auto* guard=reinterpret_cast<PrismPhysicalGuardV1*>(ptr);auto s=static_cast<cudaStream_t>(stream);
    prism_physical_guard_evaluate_v1<<<1,256,0,s>>>(guard);
    auto rc=cudaGetLastError();if(rc!=cudaSuccess)return static_cast<int>(rc);
    prism_physical_guard_transition_v1<<<1,256,0,s>>>(guard);return static_cast<int>(cudaGetLastError());
}
__global__ void prism_physical_guard_host_ceiling_v1(PrismPhysicalGuardV1* guard,uint32_t ceiling) {
    if(threadIdx.x||blockIdx.x)return;
    if(!prism_guard_owner_valid(guard)||ceiling>PRISM_GEAR_ABORT){asm volatile("trap;");return;}
    guard->snapshot.host_ceiling=ceiling;
}
extern "C" int prism_physical_guard_publish_ceiling_v1(uint64_t ptr,uint32_t ceiling,void* stream) {
    if(!ptr||!stream||ceiling>PRISM_GEAR_ABORT)return static_cast<int>(cudaErrorInvalidValue);
    prism_physical_guard_host_ceiling_v1<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(reinterpret_cast<PrismPhysicalGuardV1*>(ptr),ceiling);
    return static_cast<int>(cudaGetLastError());
}
extern "C" int prism_physical_guard_snapshot_v1(uint64_t ptr,void* stream,uint32_t timeout_ms,PrismPhysicalGuardSnapshotV1* out) {
    if(!ptr||!stream||!out||!timeout_ms)return static_cast<int>(cudaErrorInvalidValue);
    auto s=static_cast<cudaStream_t>(stream);uint64_t waited=0;
    if(prism_gear_governor_stream_capturing(s))return static_cast<int>(cudaErrorStreamCaptureUnsupported);
    int rc=prism_gearbox_bounded_drain(s,static_cast<uint64_t>(timeout_ms)*1000,&waited);if(rc)return rc;
    return static_cast<int>(cudaMemcpy(out,reinterpret_cast<void*>(ptr),sizeof(*out),cudaMemcpyDeviceToHost));
}
// Setup only. Same-context owner stream is drained before publication; no
// graph can observe partially bound storage. Full scheduled horizon includes
// terminal continuation beyond a condition's nominal protocol total.
extern "C" int prism_physical_guard_bind_time_v1(uint64_t ptr,const uint32_t* step,uint64_t capacity,void* stream) {
    if(!ptr||!step||!capacity||capacity>SIZE_MAX/(sizeof(PrismPhysicalTimeRecordV1)+sizeof(PrismPhysicalEnergyBalanceRecordV1))||!stream)
        return static_cast<int>(cudaErrorInvalidValue);
    auto s=static_cast<cudaStream_t>(stream);
    if(prism_gear_governor_stream_capturing(s))return static_cast<int>(cudaErrorStreamCaptureUnsupported);
    uint64_t waited=0;int err=prism_gearbox_bounded_drain(s,30000000,&waited);if(err)return err;
    PrismPhysicalGuardV1 h{};uint32_t initial_step=0;
    auto rc=cudaMemcpy(&h,reinterpret_cast<void*>(ptr),sizeof(h),cudaMemcpyDeviceToHost);if(rc!=cudaSuccess)return static_cast<int>(rc);
    rc=cudaMemcpy(&initial_step,step,sizeof(initial_step),cudaMemcpyDeviceToHost);if(rc!=cudaSuccess)return static_cast<int>(rc);
    if(h.time_rows||h.energy_rows||h.snapshot.physical_eval_epoch||h.last_begin_epoch||initial_step)
        return static_cast<int>(cudaErrorInvalidValue);
    rc=cudaMalloc(reinterpret_cast<void**>(&h.time_rows),static_cast<size_t>(capacity)*(sizeof(*h.time_rows)+sizeof(*h.energy_rows)));
    if(rc!=cudaSuccess)return static_cast<int>(rc);
    // One owning allocation and one rollback/free path. energy_rows is a typed
    // tail alias, never independently freed; both row types have 8-byte alignment.
    h.energy_rows=reinterpret_cast<PrismPhysicalEnergyBalanceRecordV1*>(h.time_rows+capacity);
    h.time_step=step;h.time_capacity=capacity;h.time_count=0;h.cumulative_time_ps=0.0;
    h.step_dt_ps=std::numeric_limits<double>::quiet_NaN();h.step_at_begin=0;
    rc=cudaMemcpy(reinterpret_cast<void*>(ptr),&h,sizeof(h),cudaMemcpyHostToDevice);
    if(rc!=cudaSuccess) {cudaFree(h.time_rows);return static_cast<int>(rc);}
    return static_cast<int>(cudaSuccess);
}
extern "C" int prism_physical_guard_read_time_v1(uint64_t ptr,uint64_t start,uint64_t count,
    PrismPhysicalTimeRecordV1* out,void* stream,uint32_t timeout_ms) {
    if(!ptr||!count||!out||!stream||!timeout_ms)return static_cast<int>(cudaErrorInvalidValue);
    auto s=static_cast<cudaStream_t>(stream);
    if(prism_gear_governor_stream_capturing(s))return static_cast<int>(cudaErrorStreamCaptureUnsupported);
    uint64_t waited=0;int err=prism_gearbox_bounded_drain(s,static_cast<uint64_t>(timeout_ms)*1000,&waited);if(err)return err;
    PrismPhysicalGuardV1 h{};
    auto rc=cudaMemcpy(&h,reinterpret_cast<void*>(ptr),sizeof(h),cudaMemcpyDeviceToHost);if(rc!=cudaSuccess)return static_cast<int>(rc);
    if(!h.time_rows||h.time_count>h.time_capacity||start>h.time_count||count>h.time_count-start)
        return static_cast<int>(cudaErrorInvalidValue);
    return static_cast<int>(cudaMemcpy(out,h.time_rows+start,static_cast<size_t>(count)*sizeof(*out),cudaMemcpyDeviceToHost));
}
extern "C" int prism_physical_guard_read_energy_v1(uint64_t ptr,uint64_t start,uint64_t count,
    PrismPhysicalEnergyBalanceRecordV1* out,void* stream,uint32_t timeout_ms) {
    if(!ptr||!count||!out||!stream||!timeout_ms)return static_cast<int>(cudaErrorInvalidValue);
    auto s=static_cast<cudaStream_t>(stream);
    if(prism_gear_governor_stream_capturing(s))return static_cast<int>(cudaErrorStreamCaptureUnsupported);
    uint64_t waited=0;int err=prism_gearbox_bounded_drain(s,static_cast<uint64_t>(timeout_ms)*1000,&waited);if(err)return err;
    PrismPhysicalGuardV1 h{};
    auto rc=cudaMemcpy(&h,reinterpret_cast<void*>(ptr),sizeof(h),cudaMemcpyDeviceToHost);if(rc!=cudaSuccess)return static_cast<int>(rc);
    if(!h.time_rows||!h.energy_rows||h.time_count!=h.snapshot.physical_eval_epoch||h.time_count>h.time_capacity||start>h.time_count||count>h.time_count-start)
        return static_cast<int>(cudaErrorInvalidValue);
    return static_cast<int>(cudaMemcpy(out,h.energy_rows+start,static_cast<size_t>(count)*sizeof(*out),cudaMemcpyDeviceToHost));
}
extern "C" int prism_physical_guard_destroy_v1(uint64_t ptr,void* stream,uint32_t timeout_ms) {
    if(!ptr||!stream||!timeout_ms)return static_cast<int>(cudaErrorInvalidValue);
    auto s=static_cast<cudaStream_t>(stream);uint64_t waited=0;
    if(prism_gear_governor_stream_capturing(s))return static_cast<int>(cudaErrorStreamCaptureUnsupported);
    int rc=prism_gearbox_bounded_drain(s,static_cast<uint64_t>(timeout_ms)*1000,&waited);if(rc)return rc;
    PrismPhysicalGuardV1 h{};
    auto status=cudaMemcpy(&h,reinterpret_cast<void*>(ptr),sizeof(h),cudaMemcpyDeviceToHost);
    if(status!=cudaSuccess)return static_cast<int>(status);
    if(h.time_rows) {status=cudaFree(h.time_rows);if(status!=cudaSuccess)return static_cast<int>(status);}
    return static_cast<int>(cudaFree(reinterpret_cast<void*>(ptr)));
}

extern "C" int prism_graph_rebind_handle_arguments_v1(cudaGraph_t,uint64_t,uint64_t,
    void* const*,const uint32_t*,uint32_t,uint32_t,uint32_t*);
extern "C" int prism_gearbox_rebind_parent_predicates_v1(cudaGraph_t graph,uint64_t old_handle,uint64_t new_handle,uint32_t* changed) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040
    void* functions[]={reinterpret_cast<void*>(prism_gearbox_predicate_bridge_kernel),reinterpret_cast<void*>(prism_gearbox_predicate_bridge_kernel_traced),reinterpret_cast<void*>(prism_gearbox_guard_bridge_kernel)};
    const uint32_t counts[]={3,4,4};
    return prism_graph_rebind_handle_arguments_v1(graph,old_handle,new_handle,functions,counts,3,0,changed);
#else
    return static_cast<int>(cudaErrorNotSupported);
#endif
}

// Exact module state transfer at an all-worker quiescent boundary.
// Caller owns the context, holds every producer and synchronizes before entry.
extern "C" size_t prism_checkpoint_gearbox_bytes() { return sizeof(d_gearbox_table) + sizeof(d_rescale_ratios); }
extern "C" int prism_checkpoint_gearbox_capture(unsigned char* bytes, size_t length, cudaStream_t stream) {
    if (!bytes || length != prism_checkpoint_gearbox_bytes()) return (int)cudaErrorInvalidValue;
    size_t offset = 0; cudaError_t rc;
    rc = cudaMemcpyFromSymbolAsync(bytes + offset, d_gearbox_table, sizeof(d_gearbox_table), 0, cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(d_gearbox_table);
    rc = cudaMemcpyFromSymbolAsync(bytes + offset, d_rescale_ratios, sizeof(d_rescale_ratios), 0, cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(d_rescale_ratios);
    return (int)cudaStreamSynchronize(stream);
}
extern "C" int prism_checkpoint_gearbox_restore(const unsigned char* bytes, size_t length, cudaStream_t stream) {
    if (!bytes || length != prism_checkpoint_gearbox_bytes()) return (int)cudaErrorInvalidValue;
    size_t offset = 0; cudaError_t rc;
    rc = cudaMemcpyToSymbolAsync(d_gearbox_table, bytes + offset, sizeof(d_gearbox_table), 0, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(d_gearbox_table);
    rc = cudaMemcpyToSymbolAsync(d_rescale_ratios, bytes + offset, sizeof(d_rescale_ratios), 0, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(d_rescale_ratios);
    return (int)cudaStreamSynchronize(stream);
}
