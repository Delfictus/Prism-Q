// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D — Coupled Chronology Record (CCR)
//
// The unified provenance record emitted by the fused Temporal / Gearbox /
// Tensor-Family decision path.  ONE record carries the whole provenance
// tuple that a later host-side join can never reconstruct:
//
//     global_md_step  ·  cumulative_simulated_time_ps  ·  gear_epoch_id
//     requested_gear  ·  authorized_gear  ·  device_measured_gear
//     dt_applied_ps   ·  stream_id · replica_id · emitter_tag
//     tensor_snapshot_id
//     + TSH demand summary  + physical-governor summary
//
// WHY THIS HEADER EXISTS
// ----------------------
// The decision kernel is a single-threaded <<<1,1>>> reduce-and-decide node
// inside the captured graph.  It must stage its verdict as a COMPACT NATIVE
// RECORD into a device ring: no JSON, no hashing, no compression, no
// filesystem work on the hot path.  That requires a C-visible struct with
// frozen byte offsets, which is what this header freezes.
//
// THREE-WAY LAYOUT CONTRACT
// -------------------------
// The offsets below are one of THREE bindings that must agree:
//   1. this header's `PRISM_CCR_OFF_*` macros (device writer),
//   2. `#[repr(C, align(64))] CoupledChronologyRecord` in
//      crates/prism-nhs/src/coupled_chronology.rs (host decoder),
//   3. the `CCR_OFF_*` constants in that same Rust module.
// (2) and (3) are pinned to each other by `offset_of!` static assertions,
// and (3) is pinned to (1) by a const-fn parse of THIS FILE's text via
// `include_str!`.  Drift in any one of the three is a COMPILE ERROR on the
// Rust side — it can never become a silently misparsed record.
//
// Modelled on the ZSTR frame-header discipline
// (crates/prism-nhs/src/zstr.rs + src/cuda/zstr_kernels.cuh), which uses the
// same frozen-offset + explicit-sentinel conventions.
//
// SENTINEL DISCIPLINE — the reason this record exists at all
// ----------------------------------------------------------
// The prior campaign's evidence is uninterpretable because a hardcoded
// `gear_id: 0` was indistinguishable from a MEASURED gear 0 — and gear 0
// means 0.5 fs, so the records silently asserted 4x-to-8x finer temporal
// resolution than the run actually had.  Therefore:
//
//   * gear 0 is a LEGITIMATE gear.  0 can NEVER mean "no measurement".
//   * every field has an out-of-band sentinel that no valid measurement can
//     take (`PRISM_CCR_*_UNMEASURED`).
//   * `transition_reason` and `emitter_tag` reserve the value 0 as their
//     "not recorded" sentinel and number every real value from 1.
//   * f64 fields use a DISTINGUISHED quiet-NaN payload, so "not measured"
//     stays separable from "measured as non-finite" (which is a physical
//     trap condition, not an absence).  The producer WRITES the constant;
//     it never arises from arithmetic, so no NaN-payload canonicalisation
//     can manufacture it.
//
// Compilation: header-only.  Include from the decision kernel's .cu.
//              nvcc -arch=sm_120 -std=c++17 ...
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include <cstddef>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

// ─── Schema version ─────────────────────────────────────────────────────────
//
// Bump ONLY when a field is added or its meaning changes.  Offsets 0..192 are
// FROZEN.  A reader that sees an unexpected version must fail loudly rather
// than misparse; the Rust decoder does exactly that.
#define PRISM_CCR_SCHEMA_VERSION 1u

// Total record size and alignment.  64-byte alignment puts each record on a
// Blackwell sm_120 L2 sector boundary, so a device ring slot write never
// straddles a cache line it does not own.
#define PRISM_CCR_RECORD_BYTES 192u
#define PRISM_CCR_RECORD_ALIGN 64u

// ─── Frozen byte offsets ────────────────────────────────────────────────────
//
// 8-byte quantities first (natural alignment, no implicit padding), then the
// 4-byte block, then explicit reserved tail.  There is NO implicit padding
// anywhere in this struct — every byte from 0 to 192 is named.

// ── Chronology core ──
#define PRISM_CCR_OFF_GLOBAL_MD_STEP        0u    // u64 — global MD step index
#define PRISM_CCR_OFF_CUM_SIM_TIME_PS       8u    // f64 — cumulative simulated ps
#define PRISM_CCR_OFF_DT_APPLIED_PS         16u   // f64 — dt ACTUALLY integrated
#define PRISM_CCR_OFF_GEAR_EPOCH_ID         24u   // u64 — monotonic gear epoch
#define PRISM_CCR_OFF_TENSOR_SNAPSHOT_ID    32u   // u64 — bound tensor snapshot

// ── TSH decision summary (scientific DEMAND — never authority) ──
#define PRISM_CCR_OFF_TSH_SALIENCE_PER_PS   40u   // f64 — salience per SIMULATED ps
#define PRISM_CCR_OFF_TSH_MEMORY_STATE      48u   // f64 — integrated memory scalar
#define PRISM_CCR_OFF_TSH_REQ_CONFIDENCE    56u   // f64 — confidence in the request
#define PRISM_CCR_OFF_OBSERVATION_CONF      64u   // f64 — anti-blindness confidence

// ── Physical governor summary (the SOLE safety authority) ──
#define PRISM_CCR_OFF_CONSTRAINT_RESIDUAL   72u   // f64 — constraint residual
#define PRISM_CCR_OFF_HAMILTONIAN_DRIFT     80u   // f64 — |dV| / |V + W_ext|
#define PRISM_CCR_OFF_FORCE_NORM            88u   // f64 — ||F||_2

// ── 4-byte block ──
#define PRISM_CCR_OFF_SCHEMA_VERSION        96u   // u32 — PRISM_CCR_SCHEMA_VERSION
#define PRISM_CCR_OFF_RECORD_FLAGS          100u  // u32 — PRISM_CCR_FLAG_*
#define PRISM_CCR_OFF_REQUESTED_GEAR        104u  // u32 — TSH demand
#define PRISM_CCR_OFF_AUTHORIZED_GEAR       108u  // u32 — min(demanded, permitted)
#define PRISM_CCR_OFF_DEVICE_MEASURED_GEAR  112u  // u32 — what actually ran
#define PRISM_CCR_OFF_TSH_ACTIVE_BRANCHES   116u  // u32 — branch bitmask
#define PRISM_CCR_OFF_PHYSICAL_GUARD_STATUS 120u  // u32 — PRISM_CCR_GUARD_*
#define PRISM_CCR_OFF_TRANSITION_REASON     124u  // u32 — PRISM_CCR_REASON_*
#define PRISM_CCR_OFF_HMR_VERIFIED          128u  // u32 — tri-state, see below
#define PRISM_CCR_OFF_STREAM_ID             132u  // u32 — producing stream
#define PRISM_CCR_OFF_REPLICA_ID            136u  // u32 — producing replica
#define PRISM_CCR_OFF_EMITTER_TAG           140u  // u32 — PRISM_CCR_EMITTER_*
#define PRISM_CCR_OFF_PUBLICATION_FENCE     144u  // u32 — 0 dirty / 1 published
#define PRISM_CCR_OFF_RESERVED_U32          148u  // u32 — reserved, write 0
#define PRISM_CCR_OFF_RECORD_SEQ            152u  // u64 — per-emitter sequence
#define PRISM_CCR_OFF_RESERVED              160u  // u32[8] — reserved, write 0

#define PRISM_CCR_RESERVED_U32_COUNT 8u

// ─── Sentinels — "not measured" is NEVER a plausible value ──────────────────

// Generic unsigned sentinels.  `stream_id`, `replica_id`,
// `tsh_active_branches` and `physical_guard_status` all have 0 as a
// LEGITIMATE value (stream 0; no branches active; no guard tripped), so
// absence must be out of band.
#define PRISM_CCR_U32_UNMEASURED 0xFFFFFFFFu
#define PRISM_CCR_U64_UNMEASURED 0xFFFFFFFFFFFFFFFFull

// Gear sentinel.  Mirrors `PRISM_GEARBOX_GEAR_UNMEASURED` (gearbox.cuh) and
// `PRISM_ZSTR_GEAR_ID_UNMEASURED` (zstr_kernels.cuh) so the three surfaces
// agree bit-for-bit on what "no gear measurement" looks like.
#define PRISM_CCR_GEAR_UNMEASURED 0xFFFFFFFFu

// Legal gear encoding (verified: __constant__ d_gearbox_table,
// src/cuda/gearbox.cu:28-33 == {0.0005, 0.0020, 0.0040, NaN} PICOSECONDS).
#define PRISM_CCR_GEAR_FINE_0P5FS   0u   // 0.0005 ps
#define PRISM_CCR_GEAR_DEFAULT_2FS  1u   // 0.0020 ps
#define PRISM_CCR_GEAR_HMR_4FS      2u   // 0.0040 ps
#define PRISM_CCR_GEAR_ABORT_TRAP   3u   // NaN — abort trap, not a timestep
#define PRISM_CCR_GEAR_MAX          3u

// f64 "not measured" sentinel.
//
// A quiet NaN carrying the ASCII payload 'P','R','S','M' (0x5052534D).  It is
// a NaN, so it can never be mistaken for a physical quantity, and its payload
// is distinctive, so it is separable from an ordinary NaN produced by a real
// numerical trap (for example a G29 non-finite force norm, which is a
// MEASUREMENT of a non-physical state and must NOT be reported as absence).
//
// The producer stores this constant directly; it never results from
// arithmetic, so hardware NaN-payload canonicalisation cannot fabricate it.
#define PRISM_CCR_F64_UNMEASURED_BITS 0x7FF800005052534Dull

// `hmr_verified` is TRI-state: the governor must be able to say "I have not
// established topology permission", which is different from "HMR is off".
#define PRISM_CCR_HMR_NOT_APPLIED  0u   // verified: topology.hmr_applied == false
#define PRISM_CCR_HMR_APPLIED      1u   // verified: topology.hmr_applied == true
#define PRISM_CCR_HMR_UNVERIFIED   0xFFFFFFFFu  // NOT established — no permission

// Publication fence.  Same discipline as `ZstrFrameHeader.completion_fence`:
// the slot is host-readable ONLY after the producer raises this, and the
// producer raises it LAST, after a system-scope threadfence.
#define PRISM_CCR_FENCE_DIRTY     0u
#define PRISM_CCR_FENCE_PUBLISHED 1u

// ─── record_flags bits ──────────────────────────────────────────────────────
//
// EMIT ON NO-CHANGE TOO.  A gap in the record stream must never be ambiguous
// between "the gear was held" and "the decision did not run".  Therefore
// PRISM_CCR_FLAG_DECISION_EXECUTED is set on EVERY record the decision path
// produces, and exactly one of GEAR_HELD / GEAR_CHANGED accompanies it.
#define PRISM_CCR_FLAG_DECISION_EXECUTED     (1u << 0)
#define PRISM_CCR_FLAG_GEAR_HELD             (1u << 1)
#define PRISM_CCR_FLAG_GEAR_CHANGED          (1u << 2)
// Gear flips apply END-OF-CHUNK when body_unroll > 1.  A decision taken
// mid-chunk is STAGED, not applied; this bit says so explicitly instead of
// letting the reader assume the authorized gear was live at this step.
#define PRISM_CCR_FLAG_STAGED_PENDING_BOUNDARY (1u << 3)
// The governor selected a conservative gear for the WHOLE CHUNK on entry.
#define PRISM_CCR_FLAG_CHUNK_ENTRY_CONSERVATIVE (1u << 4)
// Anti-blindness: this record is a FORCED high-resolution sentinel sample
// taken because the maximum interval without one had elapsed.
#define PRISM_CCR_FLAG_SENTINEL_SAMPLE       (1u << 5)
// Anti-blindness: observation confidence is below threshold, so coarsening
// is prohibited regardless of salience.
#define PRISM_CCR_FLAG_OBSERVATION_STARVED   (1u << 6)
// The chronology clock did NOT advance for this record because
// `dt_applied_ps` was unmeasured.  A physical clock cannot be advanced by an
// unmeasured timestep; the alternative (advancing by the AUTHORIZED dt) would
// fabricate elapsed physics.
#define PRISM_CCR_FLAG_CLOCK_STALLED         (1u << 7)
// Uncertainty was grown for this interval because evidence was sparse.
#define PRISM_CCR_FLAG_UNCERTAINTY_GROWN     (1u << 8)
// The abort trap (gear 3) fired.
#define PRISM_CCR_FLAG_ABORT_TRAP            (1u << 9)

// ─── physical_guard_status bits ─────────────────────────────────────────────
//
// A SET bit means the guard is ASSERTING — it constrains or forbids
// coarsening.  0 therefore means "every guard clear", which is a legitimate
// measured state; absence of measurement is PRISM_CCR_U32_UNMEASURED.
//
// Bit order is the LEXICOGRAPHIC decision order from the architecture
// directive.  It is deliberately NOT a weighted score: a weighted score
// always admits a salience value large enough to outvote a physical
// prohibition.  Lower bit index == higher priority.
#define PRISM_CCR_GUARD_PHYSICAL_EMERGENCY   (1u << 0)  // 1. highest priority
#define PRISM_CCR_GUARD_TOPOLOGY_FORBIDS     (1u << 1)  // 2. topology_permission
#define PRISM_CCR_GUARD_CONSTRAINT_UNSTABLE  (1u << 2)  // 3. constraint_stability
#define PRISM_CCR_GUARD_ENERGY_DRIFT         (1u << 3)  // 4. energy_drift
#define PRISM_CCR_GUARD_FORCE_UNSTABLE       (1u << 4)  // 5. force_stability
#define PRISM_CCR_GUARD_TSH_MEMORY           (1u << 5)  // 6. tsh_memory
#define PRISM_CCR_GUARD_TSH_SALIENCE         (1u << 6)  // 7. tsh_salience
#define PRISM_CCR_GUARD_PHASE_REQUIREMENT    (1u << 7)  // 8. phase_requirement
#define PRISM_CCR_GUARD_DWELL_UNSATISFIED    (1u << 8)  // 9. current_gear dwell
#define PRISM_CCR_GUARD_OBSERVATION_CONF     (1u << 9)  // anti-blindness floor
#define PRISM_CCR_GUARD_INSTABILITY_MEMORY   (1u << 10) // instability memory

// Number of defined guard bits — the Rust mirror asserts on this.
#define PRISM_CCR_GUARD_BIT_COUNT 11u

// ─── transition_reason values ───────────────────────────────────────────────
//
// 0 is RESERVED as "not recorded".  Every real reason is >= 1, so a
// zero-initialised buffer can never masquerade as a recorded decision.
#define PRISM_CCR_REASON_NOT_RECORDED                0u
#define PRISM_CCR_REASON_INITIALIZATION              1u
#define PRISM_CCR_REASON_HELD_NO_CHANGE              2u
// Finer resolution MAY be triggered by salience alone.
#define PRISM_CCR_REASON_DOWNSHIFT_SALIENCE          3u
// The governor forced finer resolution for physical safety.
#define PRISM_CCR_REASON_DOWNSHIFT_PHYSICAL_GUARD    4u
// Coarser resolution granted: low salience AND verified physical safety.
#define PRISM_CCR_REASON_UPSHIFT_GRANTED             5u
// Coarser demanded, governor refused.
#define PRISM_CCR_REASON_UPSHIFT_DENIED_GUARD        6u
// Coarser demanded, refused because observation confidence is below floor.
#define PRISM_CCR_REASON_UPSHIFT_DENIED_OBSERVATION  7u
// Coarser demanded, refused because the dwell requirement is unsatisfied.
#define PRISM_CCR_REASON_UPSHIFT_DENIED_DWELL        8u
// Topology permission ceiling bound the authorized gear.
#define PRISM_CCR_REASON_TOPOLOGY_CEILING            9u
// Conservative gear chosen for the whole chunk on entry.
#define PRISM_CCR_REASON_CHUNK_ENTRY_CONSERVATIVE    10u
// Forced high-resolution anti-blindness sentinel sample.
#define PRISM_CCR_REASON_SENTINEL_SAMPLE_FORCED      11u
// Abort trap: within-chunk emergency, gear 3.
#define PRISM_CCR_REASON_ABORT_TRAP                  12u
// Decision staged, awaiting a validated integration boundary.
#define PRISM_CCR_REASON_STAGED_PENDING_BOUNDARY     13u

#define PRISM_CCR_REASON_MAX 13u

// ─── emitter_tag values ─────────────────────────────────────────────────────
//
// UV / LIF / EFP / LADD carry DIFFERENT clocks in the same file today; the
// tag is what makes a merged in-memory stream still attributable.  0 is
// RESERVED as "unspecified".
#define PRISM_CCR_EMITTER_UNSPECIFIED      0u
#define PRISM_CCR_EMITTER_GEARBOX_DECISION 1u
#define PRISM_CCR_EMITTER_TENSOR_SNAPSHOT  2u
#define PRISM_CCR_EMITTER_UV               3u
#define PRISM_CCR_EMITTER_LIF              4u
#define PRISM_CCR_EMITTER_EFP              5u
#define PRISM_CCR_EMITTER_LADD             6u
#define PRISM_CCR_EMITTER_COFIRE           7u
#define PRISM_CCR_EMITTER_ZSTR_FRAME       8u
#define PRISM_CCR_EMITTER_GHOST_SEGMENT    9u
#define PRISM_CCR_EMITTER_SPIKE_ARROW      10u

#define PRISM_CCR_EMITTER_MAX 10u

// ─── The record ─────────────────────────────────────────────────────────────

struct alignas(PRISM_CCR_RECORD_ALIGN) PrismCoupledChronologyRecord {
    // ── chronology core ──
    uint64_t global_md_step;               // @0
    double   cumulative_simulated_time_ps; // @8
    double   dt_applied_ps;                // @16
    uint64_t gear_epoch_id;                // @24
    uint64_t tensor_snapshot_id;           // @32

    // ── TSH demand summary ──
    double   tsh_integrated_salience_per_ps; // @40  NORMALISED PER SIMULATED ps
    double   tsh_memory_state;               // @48
    double   tsh_request_confidence;         // @56
    double   observation_confidence;         // @64

    // ── physical governor summary ──
    double   constraint_residual;          // @72
    double   hamiltonian_drift;            // @80
    double   force_norm;                   // @88

    // ── 4-byte block ──
    uint32_t schema_version;               // @96
    uint32_t record_flags;                 // @100
    uint32_t requested_gear;               // @104
    uint32_t authorized_gear;              // @108
    uint32_t device_measured_gear;         // @112
    uint32_t tsh_active_branches;          // @116
    uint32_t physical_guard_status;        // @120
    uint32_t transition_reason;            // @124
    uint32_t hmr_verified;                 // @128
    uint32_t stream_id;                    // @132
    uint32_t replica_id;                   // @136
    uint32_t emitter_tag;                  // @140
    uint32_t publication_fence;            // @144
    uint32_t _reserved_u32;                // @148

    // ── loss detection ──
    //
    // Per-emitter monotonic record sequence, stamped by the producer and
    // incremented by exactly 1 on every emitted record.
    //
    // WHY A SEQUENCE IS REQUIRED, given the record already carries
    // cumulative simulated time: the physical-time continuity identity
    // `cum[i] - cum[i-1] == dt_applied[i] * (step[i] - step[i-1])` detects a
    // lost record ONLY when the gear changed across the loss. While the gear
    // is HELD — which is the common case — a dropped record leaves the
    // identity exactly satisfied, because the survivors' step gap and the
    // constant dt still multiply out correctly. Continuity is a physics
    // check, not a completeness check. The sequence is the completeness
    // check, and the two are kept as separate conservation terms.
    uint64_t record_seq;                   // @152
    uint32_t _reserved[PRISM_CCR_RESERVED_U32_COUNT]; // @160 .. @192
};

static_assert(sizeof(PrismCoupledChronologyRecord) == PRISM_CCR_RECORD_BYTES,
              "PrismCoupledChronologyRecord MUST be 192 bytes.");
static_assert(alignof(PrismCoupledChronologyRecord) == PRISM_CCR_RECORD_ALIGN,
              "PrismCoupledChronologyRecord MUST be 64-byte aligned.");

#define PRISM_CCR_ASSERT_OFF(field, macro)                                    \
    static_assert(offsetof(PrismCoupledChronologyRecord, field) == (macro),   \
                  "CCR offset drift: " #field)

PRISM_CCR_ASSERT_OFF(global_md_step,               PRISM_CCR_OFF_GLOBAL_MD_STEP);
PRISM_CCR_ASSERT_OFF(cumulative_simulated_time_ps, PRISM_CCR_OFF_CUM_SIM_TIME_PS);
PRISM_CCR_ASSERT_OFF(dt_applied_ps,                PRISM_CCR_OFF_DT_APPLIED_PS);
PRISM_CCR_ASSERT_OFF(gear_epoch_id,                PRISM_CCR_OFF_GEAR_EPOCH_ID);
PRISM_CCR_ASSERT_OFF(tensor_snapshot_id,           PRISM_CCR_OFF_TENSOR_SNAPSHOT_ID);
PRISM_CCR_ASSERT_OFF(tsh_integrated_salience_per_ps, PRISM_CCR_OFF_TSH_SALIENCE_PER_PS);
PRISM_CCR_ASSERT_OFF(tsh_memory_state,             PRISM_CCR_OFF_TSH_MEMORY_STATE);
PRISM_CCR_ASSERT_OFF(tsh_request_confidence,       PRISM_CCR_OFF_TSH_REQ_CONFIDENCE);
PRISM_CCR_ASSERT_OFF(observation_confidence,       PRISM_CCR_OFF_OBSERVATION_CONF);
PRISM_CCR_ASSERT_OFF(constraint_residual,          PRISM_CCR_OFF_CONSTRAINT_RESIDUAL);
PRISM_CCR_ASSERT_OFF(hamiltonian_drift,            PRISM_CCR_OFF_HAMILTONIAN_DRIFT);
PRISM_CCR_ASSERT_OFF(force_norm,                   PRISM_CCR_OFF_FORCE_NORM);
PRISM_CCR_ASSERT_OFF(schema_version,               PRISM_CCR_OFF_SCHEMA_VERSION);
PRISM_CCR_ASSERT_OFF(record_flags,                 PRISM_CCR_OFF_RECORD_FLAGS);
PRISM_CCR_ASSERT_OFF(requested_gear,               PRISM_CCR_OFF_REQUESTED_GEAR);
PRISM_CCR_ASSERT_OFF(authorized_gear,              PRISM_CCR_OFF_AUTHORIZED_GEAR);
PRISM_CCR_ASSERT_OFF(device_measured_gear,         PRISM_CCR_OFF_DEVICE_MEASURED_GEAR);
PRISM_CCR_ASSERT_OFF(tsh_active_branches,          PRISM_CCR_OFF_TSH_ACTIVE_BRANCHES);
PRISM_CCR_ASSERT_OFF(physical_guard_status,        PRISM_CCR_OFF_PHYSICAL_GUARD_STATUS);
PRISM_CCR_ASSERT_OFF(transition_reason,            PRISM_CCR_OFF_TRANSITION_REASON);
PRISM_CCR_ASSERT_OFF(hmr_verified,                 PRISM_CCR_OFF_HMR_VERIFIED);
PRISM_CCR_ASSERT_OFF(stream_id,                    PRISM_CCR_OFF_STREAM_ID);
PRISM_CCR_ASSERT_OFF(replica_id,                   PRISM_CCR_OFF_REPLICA_ID);
PRISM_CCR_ASSERT_OFF(emitter_tag,                  PRISM_CCR_OFF_EMITTER_TAG);
PRISM_CCR_ASSERT_OFF(publication_fence,            PRISM_CCR_OFF_PUBLICATION_FENCE);
PRISM_CCR_ASSERT_OFF(_reserved_u32,                PRISM_CCR_OFF_RESERVED_U32);
PRISM_CCR_ASSERT_OFF(record_seq,                   PRISM_CCR_OFF_RECORD_SEQ);
PRISM_CCR_ASSERT_OFF(_reserved,                    PRISM_CCR_OFF_RESERVED);

#undef PRISM_CCR_ASSERT_OFF

// ─── Device/host helpers ────────────────────────────────────────────────────
//
// These are the ONLY writers the hot path needs.  No JSON, no hashing, no
// compression, no filesystem work: a memset-equivalent field fill and a
// fenced fence-raise.

#if defined(__CUDACC__)
#define PRISM_CCR_HD __host__ __device__ __forceinline__
#else
#define PRISM_CCR_HD inline
#endif

/// Reinterpret the f64 unmeasured sentinel without invoking arithmetic.
///
/// This project compiles device code with `--use_fast_math`, which permits
/// flush-to-zero and relaxed NaN handling.  Neither can affect this sentinel:
/// the value is moved as BITS through a union and compared as an INTEGER, so
/// no floating-point operation ever touches it.  Had the sentinel been
/// produced or tested arithmetically (`0.0/0.0`, `v != v`), fast-math could
/// legally have rewritten it and collapsed "not measured" into a plain NaN.
PRISM_CCR_HD double prism_ccr_f64_unmeasured() {
    uint64_t bits = PRISM_CCR_F64_UNMEASURED_BITS;
    double out;
    // memcpy-free type pun that is well defined on both compilers used here.
    union { uint64_t u; double d; } punner;
    punner.u = bits;
    out = punner.d;
    return out;
}

/// True iff `v` carries the exact "not measured" payload.  A different NaN
/// (a real numerical trap) returns FALSE — absence and non-physicality are
/// distinct facts.
PRISM_CCR_HD bool prism_ccr_f64_is_unmeasured(double v) {
    union { uint64_t u; double d; } punner;
    punner.d = v;
    return punner.u == PRISM_CCR_F64_UNMEASURED_BITS;
}

/// Fill every field with its "not measured" sentinel and stamp the schema
/// version.  A producer MUST call this before writing the fields it actually
/// measured, so an unwritten field reads as an explicit absence rather than
/// as a zero that looks like a measurement.
PRISM_CCR_HD void prism_ccr_reset_unmeasured(PrismCoupledChronologyRecord* r) {
    if (r == nullptr) { return; }
    const double f64_un = prism_ccr_f64_unmeasured();

    r->global_md_step               = PRISM_CCR_U64_UNMEASURED;
    r->cumulative_simulated_time_ps = f64_un;
    r->dt_applied_ps                = f64_un;
    r->gear_epoch_id                = PRISM_CCR_U64_UNMEASURED;
    r->tensor_snapshot_id           = PRISM_CCR_U64_UNMEASURED;

    r->tsh_integrated_salience_per_ps = f64_un;
    r->tsh_memory_state               = f64_un;
    r->tsh_request_confidence         = f64_un;
    r->observation_confidence         = f64_un;

    r->constraint_residual = f64_un;
    r->hamiltonian_drift   = f64_un;
    r->force_norm          = f64_un;

    r->schema_version        = PRISM_CCR_SCHEMA_VERSION;
    r->record_flags          = 0u;
    r->requested_gear        = PRISM_CCR_GEAR_UNMEASURED;
    r->authorized_gear       = PRISM_CCR_GEAR_UNMEASURED;
    r->device_measured_gear  = PRISM_CCR_GEAR_UNMEASURED;
    r->tsh_active_branches   = PRISM_CCR_U32_UNMEASURED;
    r->physical_guard_status = PRISM_CCR_U32_UNMEASURED;
    r->transition_reason     = PRISM_CCR_REASON_NOT_RECORDED;
    r->hmr_verified          = PRISM_CCR_HMR_UNVERIFIED;
    r->stream_id             = PRISM_CCR_U32_UNMEASURED;
    r->replica_id            = PRISM_CCR_U32_UNMEASURED;
    r->emitter_tag           = PRISM_CCR_EMITTER_UNSPECIFIED;
    r->publication_fence     = PRISM_CCR_FENCE_DIRTY;
    r->_reserved_u32         = 0u;
    r->record_seq            = PRISM_CCR_U64_UNMEASURED;
    for (uint32_t i = 0; i < PRISM_CCR_RESERVED_U32_COUNT; ++i) {
        r->_reserved[i] = 0u;
    }
}

#if defined(__CUDACC__)
/// Publish a staged record.  System-scope fence FIRST so every field store is
/// visible to the host before the fence word flips — identical discipline to
/// `zstr_signal_completion_kernel`.
__device__ __forceinline__ void prism_ccr_publish(PrismCoupledChronologyRecord* r) {
    if (r == nullptr) { return; }
    __threadfence_system();
    *((volatile uint32_t*)&r->publication_fence) = PRISM_CCR_FENCE_PUBLISHED;
}
#endif
