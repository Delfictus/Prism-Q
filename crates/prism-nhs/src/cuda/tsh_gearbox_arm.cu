// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D — GEARBOX-FACING TSH ARM: COMPILED TRANSLATION UNIT
// Package 4 of the Coupled Temporal / Gearbox / Tensor-Family architecture.
// ═══════════════════════════════════════════════════════════════════════════
//
// WHY THIS FILE EXISTS
//
//   `tsh_gearbox_arm.cuh` is header-only.  A header that no translation unit
//   includes is compiled by NOTHING: its `static_assert`s never execute, its
//   struct layouts are never checked, and a size or offset drift would be
//   discovered at the first misparse of a science record rather than at build
//   time.  The header carried fifteen `static_assert`s and a comment claiming
//   the layouts had been "verified by compiling the header and printing
//   sizeof" — a claim nothing in the tree could reproduce.  That is the same
//   built-but-never-activated defect this package exists to repair, one layer
//   down.
//
//   This .cu is registered in `crates/prism-nhs/build.rs`, so:
//
//     * every `static_assert` in the header EXECUTES on every build, and a
//       layout drift is a compile error;
//     * `prism_tsh_arm_abi_sizes` reports the sizes the compiler ACTUALLY
//       produced, which the Rust test `device_abi_sizes_match_the_compiled_header`
//       compares against `DEVICE_*_BYTES`;
//     * the Rust parity test drives THIS implementation through the host-
//       callable entry points below and compares it against the host oracle,
//       instead of comparing the oracle against literals transcribed into a
//       comment.
//
// COMPILATION MODE
//
//   Registered through `compile_to_static_archive_ieee754`, i.e. WITHOUT
//   `--use_fast_math`, for the same reason `deterministic_reduce.cu` is:
//   everything in here is a CONTROL DECISION input, not a physics result.
//   Fast math contracts `a + b*c` into an FMA and flushes denormals, both of
//   which are deterministic on-device but no longer reproducible by the host
//   reference — and the host reference is what an offline reader uses to
//   re-derive why a gear changed.
//
// AUTHORITY
//
//   Nothing here authorizes a gear.  There is no write to a timestep, no write
//   to an authorized gear, and no call into the gearbox governor: the entry
//   points below evaluate the arm and produce a REQUEST.  The Rust test
//   `arm_source_contains_no_authorization_path` reads this file and the header
//   and fails if that ever stops being true.
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>

#include "tsh_gearbox_arm.cuh"

// ───────────────────────────────────────────────────────────────────────────
// Layout report
// ───────────────────────────────────────────────────────────────────────────

/// Slot order of the sizes reported by `prism_tsh_arm_abi_sizes`.
#define PRISM_TSH_ARM_ABI_SLOT_BRANCH  0
#define PRISM_TSH_ARM_ABI_SLOT_POLICY  1
#define PRISM_TSH_ARM_ABI_SLOT_STATE   2
#define PRISM_TSH_ARM_ABI_SLOT_RECORD  3
#define PRISM_TSH_ARM_ABI_SLOT_RING    4
/// Offset of the branch array inside `PrismTshArmState`.
#define PRISM_TSH_ARM_ABI_SLOT_STATE_BRANCH_OFFSET 5
/// Offset of the first of the fourteen contiguous policy scalars.
#define PRISM_TSH_ARM_ABI_SLOT_POLICY_SCALAR_OFFSET 6
#define PRISM_TSH_ARM_ABI_SLOT_COUNT 7

/// Report the struct sizes and the two offsets the host encoder depends on.
///
/// These are `sizeof`/`offsetof` as the compiler computed them — not constants
/// re-declared here, which would only prove this file agrees with itself.
extern "C" uint32_t prism_tsh_arm_abi_sizes(uint64_t* out, uint32_t capacity) {
    if (out == nullptr || capacity < (uint32_t)PRISM_TSH_ARM_ABI_SLOT_COUNT) {
        return 0u;
    }
    out[PRISM_TSH_ARM_ABI_SLOT_BRANCH] = (uint64_t)sizeof(PrismTshArmBranchState);
    out[PRISM_TSH_ARM_ABI_SLOT_POLICY] = (uint64_t)sizeof(PrismTshArmPolicy);
    out[PRISM_TSH_ARM_ABI_SLOT_STATE]  = (uint64_t)sizeof(PrismTshArmState);
    out[PRISM_TSH_ARM_ABI_SLOT_RECORD] = (uint64_t)sizeof(PrismTshArmRecord);
    out[PRISM_TSH_ARM_ABI_SLOT_RING]   = (uint64_t)sizeof(PrismTshArmRing);
    out[PRISM_TSH_ARM_ABI_SLOT_STATE_BRANCH_OFFSET] =
        (uint64_t)offsetof(PrismTshArmState, branch);
    out[PRISM_TSH_ARM_ABI_SLOT_POLICY_SCALAR_OFFSET] =
        (uint64_t)offsetof(PrismTshArmPolicy, branch_tau_ps);
    return (uint32_t)PRISM_TSH_ARM_ABI_SLOT_COUNT;
}

/// Schema version this build speaks.  A reader that sees another value must
/// fail loudly rather than misparse.
extern "C" uint32_t prism_tsh_arm_schema_version(void) {
    return PRISM_TSH_ARM_SCHEMA_VERSION;
}

// ───────────────────────────────────────────────────────────────────────────
// Host-callable entry points
// ───────────────────────────────────────────────────────────────────────────
//
// Every function in the header is `__host__ __device__`, so the SAME code the
// captured graph runs is reachable from the host without a GPU.  That is what
// lets the parity test compare implementations rather than comments, and it is
// why these wrappers take `void*`: the Rust side owns byte buffers whose
// layout the assertions above have already pinned.

/// Fill `policy` with the documented defaults, UNARMED.
extern "C" void prism_tsh_arm_host_default_policy(void* policy) {
    prism_tsh_arm_default_policy(static_cast<PrismTshArmPolicy*>(policy));
}

/// Reset `state` to its pre-decision baseline at simulated time `t0_ps`.
extern "C" void prism_tsh_arm_host_reset(void* state, double t0_ps) {
    prism_tsh_arm_reset(static_cast<PrismTshArmState*>(state), t0_ps);
}

/// Stage one branch observation.  Returns 1 when staged, 0 when refused.
extern "C" uint32_t prism_tsh_arm_host_observe(const void* policy,
                                               void*       state,
                                               uint32_t    branch,
                                               double      accumulated_activity,
                                               double      span_ps,
                                               double      window_end_ps,
                                               uint32_t    sample_count) {
    return prism_tsh_arm_observe(static_cast<const PrismTshArmPolicy*>(policy),
                                 static_cast<PrismTshArmState*>(state),
                                 branch,
                                 accumulated_activity,
                                 span_ps,
                                 window_end_ps,
                                 sample_count);
}

/// Take one decision.  `out_req` and `out_rec` may be null.
extern "C" void prism_tsh_arm_host_decide(const void* policy,
                                          void*       state,
                                          double      now_ps,
                                          uint64_t    global_md_step,
                                          uint32_t    frame,
                                          uint32_t    authorized_gear,
                                          uint32_t    authorized_gear_valid,
                                          void*       out_req,
                                          void*       out_rec) {
    prism_tsh_arm_decide(static_cast<const PrismTshArmPolicy*>(policy),
                         static_cast<PrismTshArmState*>(state),
                         now_ps,
                         global_md_step,
                         frame,
                         authorized_gear,
                         authorized_gear_valid,
                         static_cast<PrismTshSalienceRequest*>(out_req),
                         static_cast<PrismTshArmRecord*>(out_rec));
}

/// Pre-fill a ring slot with the UNWRITTEN pattern.
///
/// The allocator calls this instead of leaving slots zeroed, because a
/// zero-filled slot decodes as a plausible calm record at gear 0 — and gear 0
/// is a LEGITIMATE gear, so the collision is not detectable after the fact.
extern "C" void prism_tsh_arm_host_record_unwritten(void* record) {
    prism_tsh_arm_record_unwritten(static_cast<PrismTshArmRecord*>(record));
}

/// Whether a ring slot holds a record an emitter actually wrote.
extern "C" uint32_t prism_tsh_arm_host_record_is_valid(const void* record) {
    return prism_tsh_arm_record_is_valid(static_cast<const PrismTshArmRecord*>(record)) ? 1u : 0u;
}

/// Per-branch observation coverage in [0,1] — anti-blindness (c) as a number.
extern "C" double prism_tsh_arm_host_branch_coverage(const void* policy,
                                                     const void* state,
                                                     uint32_t    branch) {
    if (policy == nullptr || state == nullptr || branch >= PRISM_TSH_ARM_N_BRANCHES) {
        return 0.0;
    }
    const PrismTshArmPolicy* p = static_cast<const PrismTshArmPolicy*>(policy);
    const PrismTshArmState*  s = static_cast<const PrismTshArmState*>(state);
    return prism_tsh_arm_branch_coverage(p, &s->branch[branch]);
}

// ───────────────────────────────────────────────────────────────────────────
// Captured-graph kernel
// ───────────────────────────────────────────────────────────────────────────

/// Single-thread device evaluation of the arm, for capture into the V2 graph.
///
/// ONE thread by construction: every reduction in the header is a fixed-order
/// serial loop over branch indices, and the ring is single-producer written by
/// the same thread that decides.  Adding threads would require an atomic
/// reduction, whose non-associativity can flip a threshold comparison between
/// two otherwise identical runs — the run would then take a different gear and
/// the provenance record would faithfully report a different outcome for the
/// same input.
///
/// The kernel produces a REQUEST and a RECORD.  It does not authorize a gear
/// and it does not write a timestep.
extern "C" __global__ void prism_tsh_arm_decide_kernel(const PrismTshArmPolicy* policy,
                                                       PrismTshArmState*        state,
                                                       PrismTshArmRing*         ring,
                                                       PrismTshSalienceRequest* out_req,
                                                       double                   now_ps,
                                                       unsigned long long       global_md_step,
                                                       uint32_t                 frame,
                                                       uint32_t                 authorized_gear,
                                                       uint32_t                 authorized_gear_valid) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) {
        return;
    }
    if (policy == nullptr || state == nullptr) {
        return;
    }
    PrismTshArmRecord rec;
    prism_tsh_arm_record_unwritten(&rec);
    prism_tsh_arm_decide(policy,
                         state,
                         now_ps,
                         (uint64_t)global_md_step,
                         frame,
                         authorized_gear,
                         authorized_gear_valid,
                         out_req,
                         &rec);
    // A full ring DROPS and COUNTS; it never blocks the decision and never
    // overwrites an unread record.  The drop raises PRISM_TSH_FLAG_RING_DROPPED
    // on every subsequent record until the host clears it, so a truncated
    // stream cannot be mistaken for a complete one.
    if (ring != nullptr) {
        (void)prism_tsh_arm_ring_push(ring, state, &rec);
    }
}

/// Launch `prism_tsh_arm_decide_kernel` on `stream`.
///
/// Provided so the Rust side never has to reproduce the launch geometry: it is
/// <<<1,1>>> for the determinism reason above, and that is not a tuning knob.
extern "C" cudaError_t prism_tsh_arm_launch_decide(const PrismTshArmPolicy* policy,
                                                   PrismTshArmState*        state,
                                                   PrismTshArmRing*         ring,
                                                   PrismTshSalienceRequest* out_req,
                                                   double                   now_ps,
                                                   unsigned long long       global_md_step,
                                                   uint32_t                 frame,
                                                   uint32_t                 authorized_gear,
                                                   uint32_t                 authorized_gear_valid,
                                                   cudaStream_t             stream) {
    prism_tsh_arm_decide_kernel<<<1, 1, 0, stream>>>(policy,
                                                     state,
                                                     ring,
                                                     out_req,
                                                     now_ps,
                                                     global_md_step,
                                                     frame,
                                                     authorized_gear,
                                                     authorized_gear_valid);
    return cudaGetLastError();
}
