// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D — PHYSICAL STABILITY GOVERNOR (Package 3) — TRANSLATION UNIT
// ═══════════════════════════════════════════════════════════════════════════
//
// WHY THIS FILE EXISTS
//
// `physical_governor.cuh` is written as `__host__ __device__ __forceinline__`,
// which made it tempting to call "header-only, no build.rs registration
// needed".  That was a defect, not a design: a `static_assert` inside a header
// that NO translation unit `#include`s is never evaluated by any compiler.  In
// that state
//
//   * `sizeof(PrismPhysGovExtPolicy) == 208`, `... ExtObservation == 112` and
//     `... Decision == 128` were unchecked on the C side, so a C-side field
//     insertion could drift silently away from the Rust `#[repr(C)]` mirror
//     (the Rust `const _: () = assert!(...)` only pins the Rust half);
//   * all six `offsetof` pins were unchecked;
//   * the eighteen-link lexicographic rank-order chain — the mechanical proof
//     that no salience can outrank a physical prohibition — was unchecked;
//   * the band-monotonicity chain was unchecked;
//   * `prism_physical_governor_decide` existed in no binary, and the header's
//     own claim "it compiles standalone under nvcc -std=c++17" had never been
//     executed by anything.
//
// This translation unit is what makes every one of those assertions BINDING.
// It is compiled on every `gpu`-feature build of `prism-nhs` by
// `crates/prism-nhs/build.rs` (see the `compile_to_static_archive_ieee754`
// call), so a drift in either half now fails the build instead of shipping.
//
// FLOATING-POINT CONTRACT
//
// Compiled through the IEEE-strict entry point, NOT the default fast-math one,
// for the same reason `deterministic_reduce.cu` is: every number this file
// produces is a CONTROL DECISION input that gets compared against a threshold
// and then recorded as the reason a gear changed.  `--use_fast_math` implies
// `-ftz=true -fmad=true -prec-div=false`; FMA contraction of `0.5f*a*dt*dt` and
// denormal flushing both perturb the value relative to the Rust host mirror,
// which would break the off-device re-derivation of a recorded decision and
// could flip a band comparison between the two halves.
//
// DEVICE CODE IS GENUINELY EMITTED
//
// `prism_physical_governor_decide_kernel` below is a real `__global__` that
// calls the evaluator, so the `__device__` half is driven through ptxas rather
// than merely parsed.  Without it, an `__forceinline__` function that nothing
// calls from device code produces no device machine code at all and "the device
// evaluator compiles" would remain an untested claim.
//
// Compilation: nvcc -std=c++17 -arch=sm_120 -O3 -fmad=false --ftz=false
//              --prec-div=true --prec-sqrt=true --restrict
//              --expt-relaxed-constexpr -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════════

#include "physical_governor.cuh"

#include <cuda_runtime.h>

namespace prism_nhs { namespace physical_governor {

// ═══════════════════════════════════════════════════════════════════════════
// DEVICE ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════

/// Single-thread evaluator node.  The governor decision is one decision per
/// captured-graph replay, not a per-atom quantity, so a 1x1 launch is the
/// correct shape — parallelising it would only create an ordering hazard on a
/// value that must be bit-reproducible.
///
/// Reads five device-resident inputs and writes one 128-byte record.  No
/// atomics, no shared memory, no allocation, no I/O: the same pure function the
/// host FFI calls, so a host unit test and the device path provably agree.
__global__ void prism_physical_governor_decide_kernel(
    const PrismGearGovernorPolicy*    base_policy,
    const PrismPhysGovExtPolicy*      ext_policy,
    const PrismGearGuardObservation*  base_observation,
    const PrismPhysGovExtObservation* ext_observation,
    const PrismTshSalienceRequest*    tsh,          /* nullable */
    PrismPhysGovDecision*             out)
{
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    if (base_policy == nullptr || ext_policy == nullptr ||
        base_observation == nullptr || ext_observation == nullptr ||
        out == nullptr) {
        return;
    }
    prism_physical_governor_decide(base_policy, ext_policy, base_observation,
                                   ext_observation, tsh, out);
}

}}  // namespace prism_nhs::physical_governor

extern "C" {

// ═══════════════════════════════════════════════════════════════════════════
// HOST ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate the composed physical stability governor on the host.
///
/// This calls the SAME `prism_physical_governor_decide` the device kernel
/// calls, so a host caller exercises the production decision path rather than a
/// parallel re-implementation of it.  It performs no CUDA work — the evaluator
/// is pure arithmetic — so it is usable in dry-run policy validation as well as
/// in the live path.
///
/// Returns `cudaSuccess` (0) on success, `cudaErrorInvalidValue` on a null
/// required pointer.  A `tsh` of `nullptr` means "no scientific opinion" and is
/// legal.
///
/// NOTE ON SCHEMA REFUSAL: an unarmed or wrong-version extension policy /
/// observation is NOT reported through the return code.  It is a governed
/// outcome, not a call error: the evaluator emits a full decision record with
/// `ext_binding_gate == PRISM_PHYSGOV_GATE_EXT_BINDING` and a named
/// `ext_binding_refusal` cause, and pins the gear to the documented unarmed
/// cap.  Every invocation therefore produces exactly one record, so a gap in
/// the record stream unambiguously means "the decision did not run".
int prism_physical_governor_decide_host(
    const PrismGearGovernorPolicy*    base_policy,
    const PrismPhysGovExtPolicy*      ext_policy,
    const PrismGearGuardObservation*  base_observation,
    const PrismPhysGovExtObservation* ext_observation,
    const PrismTshSalienceRequest*    tsh,
    PrismPhysGovDecision*             out)
{
    if (base_policy == nullptr || ext_policy == nullptr ||
        base_observation == nullptr || ext_observation == nullptr ||
        out == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    prism_physical_governor_decide(base_policy, ext_policy, base_observation,
                                   ext_observation, tsh, out);
    return static_cast<int>(cudaSuccess);
}

/// Fill `out` with the documented extension-policy defaults.
///
/// The `PRISM_PHYSGOV_DEFAULT_*` macro block in the header is the single source
/// of default truth; this exposes it to the host so the Rust mirror can be
/// cross-checked against the values the DEVICE would actually use rather than
/// against a transcription of them.
///
/// `armed` is left at 0 deliberately: the host must resolve and arm a policy
/// explicitly, so an unarmed governor is never mistaken for a governed one.
int prism_physgov_default_ext_policy(PrismPhysGovExtPolicy* out) {
    if (out == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    prism_physgov_fill_default_ext_policy(out);
    return static_cast<int>(cudaSuccess);
}

/// Launch the device evaluator on `stream`.
///
/// All six pointers must be DEVICE-resident (or null, for `tsh`).  This is the
/// captured-graph-safe form: a single 1x1 kernel node with no host bridge, so
/// it can be recorded into the V2 graph alongside the gearbox pointer-swap
/// kernel.
int prism_physical_governor_decide_device(
    const PrismGearGovernorPolicy*    d_base_policy,
    const PrismPhysGovExtPolicy*      d_ext_policy,
    const PrismGearGuardObservation*  d_base_observation,
    const PrismPhysGovExtObservation* d_ext_observation,
    const PrismTshSalienceRequest*    d_tsh,
    PrismPhysGovDecision*             d_out,
    void*                             stream)
{
    if (d_base_policy == nullptr || d_ext_policy == nullptr ||
        d_base_observation == nullptr || d_ext_observation == nullptr ||
        d_out == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    prism_nhs::physical_governor::prism_physical_governor_decide_kernel
        <<<1, 1, 0, s>>>(d_base_policy, d_ext_policy, d_base_observation,
                         d_ext_observation, d_tsh, d_out);
    return static_cast<int>(cudaGetLastError());
}

// ═══════════════════════════════════════════════════════════════════════════
// ABI PROBE
//
// The sizes below are MEASURED BY THE COMPILER THAT BUILT THIS ARCHIVE, not
// transcribed from a comment.  The Rust side calls this and asserts against its
// own `size_of`, which regenerates the "verified by compiling the header and
// printing sizeof" claim from the actual build on every test run instead of
// trusting a hand-copied number.
// ═══════════════════════════════════════════════════════════════════════════

/// Link probe.  Sentinel 0x50475631 ("PGV1" — Physical GoVernor v1).  Pinned by
/// the Rust-side test; a non-matching value means the wrong archive linked.
uint32_t prism_physgov_link_probe(void) {
    return 0x50475631u;
}

/// C-side `sizeof` for the three wire structs and the schema version.
/// Every pointer is optional; nulls are skipped.
void prism_physgov_abi_sizes(
    uint32_t* ext_policy_bytes,
    uint32_t* ext_observation_bytes,
    uint32_t* decision_bytes,
    uint32_t* schema_version)
{
    if (ext_policy_bytes      != nullptr) *ext_policy_bytes      = (uint32_t)sizeof(PrismPhysGovExtPolicy);
    if (ext_observation_bytes != nullptr) *ext_observation_bytes = (uint32_t)sizeof(PrismPhysGovExtObservation);
    if (decision_bytes        != nullptr) *decision_bytes        = (uint32_t)sizeof(PrismPhysGovDecision);
    if (schema_version        != nullptr) *schema_version        = PRISM_PHYSGOV_SCHEMA_VERSION;
}

/// C-side `offsetof` for the fields the header pins, in the order
/// `{ExtPolicy.force_rate_hold_per_ps, ExtPolicy.phase_ceiling_gear,
///   ExtPolicy.ext_policy_digest, ExtObservation.hmr_applied_verified,
///   ExtObservation.max_accel_a_per_ps2, ExtObservation.tsh_instability_memory,
///   Decision.ext_binding_gate, Decision.downshift_latency_steps,
///   Decision.policy_digest, Decision.ext_binding_refusal}`.
///
/// `out` must have room for `PRISM_PHYSGOV_ABI_OFFSET_COUNT` entries.  Returns
/// the number written.
uint32_t prism_physgov_abi_offsets(uint32_t* out, uint32_t capacity) {
    const uint32_t values[] = {
        (uint32_t)offsetof(PrismPhysGovExtPolicy, force_rate_hold_per_ps),
        (uint32_t)offsetof(PrismPhysGovExtPolicy, phase_ceiling_gear),
        (uint32_t)offsetof(PrismPhysGovExtPolicy, ext_policy_digest),
        (uint32_t)offsetof(PrismPhysGovExtObservation, hmr_applied_verified),
        (uint32_t)offsetof(PrismPhysGovExtObservation, max_accel_a_per_ps2),
        (uint32_t)offsetof(PrismPhysGovExtObservation, tsh_instability_memory),
        (uint32_t)offsetof(PrismPhysGovDecision, ext_binding_gate),
        (uint32_t)offsetof(PrismPhysGovDecision, downshift_latency_steps),
        (uint32_t)offsetof(PrismPhysGovDecision, policy_digest),
        (uint32_t)offsetof(PrismPhysGovDecision, ext_binding_refusal),
    };
    const uint32_t n = (uint32_t)(sizeof(values) / sizeof(values[0]));
    if (out == nullptr) return n;
    const uint32_t limit = (capacity < n) ? capacity : n;
    for (uint32_t i = 0u; i < limit; ++i) out[i] = values[i];
    return limit;
}

}  // extern "C"

// ═══════════════════════════════════════════════════════════════════════════
// TRANSLATION-UNIT-LOCAL SELF-CHECKS
//
// The header's own `static_assert`s fire by virtue of the `#include` above.
// These add the checks that only make sense once something is actually
// compiling this: that the C++ compiler agrees with the Rust mirror on the
// constants the FFI boundary depends on.
// ═══════════════════════════════════════════════════════════════════════════

static_assert(PRISM_PHYSGOV_SCHEMA_VERSION == 1u,
              "the Rust mirror pins PHYSGOV_SCHEMA_VERSION = 1");
static_assert(PRISM_PHYSGOV_GATE_EXT_BINDING == 39u,
              "the Rust mirror pins GATE_EXT_BINDING = 39");
static_assert(PRISM_PHYSGOV_EMERG_TEMPERATURE_RUNAWAY == 0x800u,
              "the Rust mirror pins EMERG_TEMPERATURE_RUNAWAY = 0x800");
static_assert(PRISM_PHYSGOV_REFUSAL_NONE == 0u &&
              PRISM_PHYSGOV_REFUSAL_UNARMED == 1u &&
              PRISM_PHYSGOV_REFUSAL_POLICY_SCHEMA == 2u &&
              PRISM_PHYSGOV_REFUSAL_OBSERVATION_SCHEMA == 3u,
              "the Rust mirror pins the REFUSAL_* codes");
static_assert(PRISM_PHYSGOV_DEFAULT_UNARMED_CAP_GEAR == PRISM_GEAR_FINE,
              "the documented unarmed cap is the finest gear (fail closed)");
