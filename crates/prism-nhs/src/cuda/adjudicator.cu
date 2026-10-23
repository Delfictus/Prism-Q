// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / Adjudicator — T1 Quantum-Photonic Bridge + T2 KL-Divergence
//                          (Blackwell sm_120, Anti-Greenfield-compliant,
//                           Total Function over all 32-bit f32 inputs)
// ═══════════════════════════════════════════════════════════════════════
//
// Two device-side primitives + one host-callable adjudicator step:
//
//   T1 — `prism_compute_quantum_weight(intensity_packed)` returns
//        Q_s = μ_01²(λ) · I, branchless via __constant__ LUT.
//
//   T2 — `prism_adjudicator_step_kernel` is a single-thread <<<1, 1, 0,
//        stream>>> launch that consumes one (relaxed, perturbed)
//        ContactShellTile pair and writes:
//        * current_divergence (Σ KL contributions)
//        * adjudication_code (0/1/2 per pre_rank::AdjudicationCode)
//        * start_clock / stop_clock (clock64 cycles for T4 telemetry)
//        * legacy_centroid_fallback (Path A: AABB midpoint of relaxed)
//
// Numerical defense (T2):
//   - C++ pre-check: `isfinite(p_raw) && isfinite(q_raw)` increments a
//     violation counter on dirty input.
//   - C++ defensive padding: `fmaxf(., epsilon)` clamps zeros / -Inf.
//     Inf is overwritten with epsilon before this if violation_count was
//     bumped, so the ratio stays finite for log evaluation.
//   - Inline PTX guard: `setp.nan.f32` + `setp.le.f32` zero out log_ratio
//     on any residual badness. Defense in depth.
//   - Final code: violation_count > 0 ⇒ Violation (2); else 3-σ threshold
//     decides Prune (0) / Construct (1).
//
// LUT derivation (T1, Anti-Greenfield-compliant):
//   μ_01²(λ) ∝ ε(λ_probe), with ε approximated via gaussian attenuation
//   around each chromophore's λ_max (per uv_bias.rs § Chromophore
//   Absorption Profiles). Normalized by ABSORPTION_NORMALIZATION
//   (= TRP_EXTINCTION_280 = 5600 from crate::config).
//
//     260 nm: PHE-dominant.
//             ε_260 ≈ PHE_EXTINCTION_280 · exp(−((260−258) / 10)²)
//                   ≈ 200 · exp(−0.04) ≈ 192.16
//             μ²    ≈ 192.16 / 5600 ≈ 0.0343
//
//     280 nm: TRP primary trigger.
//             ε_280 = TRP_EXTINCTION_280 = 5600
//             μ²    = 5600 / 5600 = 1.0000
//
//     305 nm: TRP gaussian tail.
//             ε_305 ≈ 5600 · exp(−((305−280) / 15)²) ≈ 5600 · 0.0621
//             μ²    ≈ 0.0621
//
//     320 nm: TRP control / baseline.
//             ε_320 ≈ 5600 · exp(−((320−280) / 15)²) ≈ 5600 · 0.000810
//             μ²    ≈ 0.000810
//
// ═══════════════════════════════════════════════════════════════════════

#include "adjudicator.cuh"  // transitively #includes so3_project.cuh (header fusion)
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>  // Amendment 3.10 — kernel-level printf triage

// ════════════════════════════════════════════════════════════════════
// __constant__ memory LUT — populated by init_constants() at startup.
// Static initializer is the canonical CUDA pattern; no init kernel
// needed. nvcc emits a module-load-time copy from .nv_const to
// constant memory when the symbol is first referenced.
// ════════════════════════════════════════════════════════════════════

__constant__ float c_mu_01_sq[PRISM_MU_01_SQ_N] = {
    0.0343f,    // 260 nm — PHE-dominant
    1.0000f,    // 280 nm — TRP primary
    0.0621f,    // 305 nm — TRP gaussian tail
    0.000810f,  // 320 nm — TRP control / baseline
};

// ════════════════════════════════════════════════════════════════════
// T1 — prism_compute_quantum_weight (out-of-line definition)
// ════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float
prism_compute_quantum_weight(uint32_t intensity_packed) {
    uint32_t uv_code = prism_extract_uv_code(intensity_packed);
    float    intensity = prism_extract_intensity(intensity_packed);
    // Branchless LDC: the wavelength code indexes into c_mu_01_sq.
    return c_mu_01_sq[uv_code & 0x3u] * intensity;
}

// ════════════════════════════════════════════════════════════════════
// T2 — KL-divergence Adjudicator kernel
// ════════════════════════════════════════════════════════════════════

namespace {

// PTX-guarded log2 — defense-in-depth: even if fmaxf-clamped inputs
// produce a finite ratio, this guard catches any residual NaN / le0.
//
// On clean input: emits `lg2.approx.f32` (≈ 4-cycle hardware op).
// On NaN: log_ratio = 0.0
// On ratio ≤ 0: log_ratio = 0.0
__device__ __forceinline__ float prism_log2_with_guard(float ratio) {
    // PTX-correct guard: PTX `setp.nan.f32` is BINARY (returns true
    // when either operand is NaN); we pass `%1, %1` so the predicate
    // fires exactly when `ratio` is NaN. PTX also allows only ONE
    // guard predicate per instruction, so the "either-or" path is
    // expressed via `or.pred` into a combined predicate.
    float log_ratio;
    asm volatile (
        "{\n\t"
        " .reg .pred p_nan, p_le0, p_bad;\n\t"
        " setp.nan.f32 p_nan, %1, %1;\n\t"
        " setp.le.f32  p_le0, %1, 0f00000000;\n\t"
        " or.pred      p_bad, p_nan, p_le0;\n\t"
        " @p_bad  mov.f32        %0, 0f00000000;\n\t"
        " @!p_bad lg2.approx.f32 %0, %1;\n\t"
        "}"
        : "=f"(log_ratio)
        : "f"(ratio)
    );
    return log_ratio;
}

}  // anonymous namespace

/// T13 — 4-plane weighted KL fusion Adjudicator (SIMT).
///
/// Launches as <<<1, 64, 0, stream>>>.  Each thread `i` < n_clusters
/// processes cluster `i`'s (relaxed, perturbed) ContactShellTile pair
/// and computes the weighted KL divergence across all 4 planes:
///
///   Δ_i = ω_G·KL_geo + ω_C·KL_caus + ω_T·KL_therm + ω_H·KL_chem
///
/// with operator-mandated weights (7C8R dimer calibration):
///   ω_G = 0.4  (Geometry — pocket presence)
///   ω_C = 0.3  (Causality — trigger lag)
///   ω_T = 0.2  (Thermodynamics — water flux)
///   ω_H = 0.1  (Chemistry — aromatic shifts)
///
/// G28 mask veto (Amendment 3.4): if `(force_prune_mask >> i) & 1`,
/// thread i forces its cluster contribution to 0 BEFORE the KL sum —
/// a vetoed cluster cannot contribute to the campaign-level decision.
///
/// Zero-Trust hard-trap (T13 §4.1): non-finite values on the Causality
/// or Thermodynamics planes raise the cluster's violation flag; the
/// block-reduced OR routes the F1 SWITCH to PRISM_ADJ_VIOLATION,
/// triggering the abort sub-graph. NaN on geometry / chemistry is
/// substituted with epsilon (less aggressive — those planes can carry
/// numerical drift without indicating non-physical force).
///
/// Reduction: 64-element shared-memory butterfly (sum for KL,
/// bitwise-OR for violation), thread 0 writes the scalar outputs.
__global__ void prism_interferometric_adjudicator_step_kernel(
    InterferometricAdjudicatorFfi* __restrict__ adjudicator,
    uint32_t                                    n_clusters
) {
    const uint32_t i = threadIdx.x;

    // Note (T4): start_clock / stop_clock are owned by the pipeline
    // bookend kernels (`prism_pipeline_clock_start_kernel` /
    // `_stop_kernel`). T2 deliberately does NOT touch them so the
    // SO(3)→T2→ASC pipeline-wide timing is not clobbered.

    const ContactShellTile* relaxed_arr   = adjudicator->relaxed_manifold_ptr;
    const ContactShellTile* perturbed_arr = adjudicator->perturbed_manifold_ptr;

    // Defensive whole-block null-pointer guard. All threads converge on
    // the same answer, no warp divergence; thread 0 writes the violation
    // and we exit cleanly. First-frame seed before manifolds populated.
    if (relaxed_arr == nullptr || perturbed_arr == nullptr) {
        if (i == 0u) {
            adjudicator->current_divergence = 0.0f;
            __threadfence();
            adjudicator->adjudication_code = PRISM_ADJ_VIOLATION;
        }
        return;
    }

    // Cap at SIMT block width (64). Excess clusters beyond 64 are NOT
    // adjudicated this frame — the captured-graph topology fixes the
    // launch geometry, so multi-block adjudication is deferred to a
    // separate refactor (operator-flagged scope cap).
    constexpr uint32_t SIMT_WIDTH = 64u;
    constexpr float    EPSILON    = 1.0e-7f;

    // 4-plane weights — pinned per the T13 specification. Mutating
    // these requires re-validation against the 4LPK / 7C8R cal runs.
    constexpr float WEIGHTS[4] = { 0.4f, 0.3f, 0.2f, 0.1f };

    // Pre-fetch the G28 prune mask once per thread (broadcast load).
    // mask == 0 when SISR is disabled (single-cluster / non-dimer);
    // mask bit i set ⇒ cluster i fails bilateral symmetry.
    const uint64_t prune_mask = (adjudicator->force_prune_mask_ptr != nullptr)
                                ? *(adjudicator->force_prune_mask_ptr)
                                : 0ull;
    const bool i_vetoed = ((prune_mask >> i) & 1ull) != 0ull;

    float    my_kl              = 0.0f;
    uint32_t my_violation       = 0u;
    // B.3.2 TIY (Total Information Yield) — operator addendum 2026-05-02:
    // track the max UNWEIGHTED per-plane KL across all 4 planes so a
    // strong Causal or Thermodynamic spike triggers Construct (Gear 0)
    // even when Geometry is quiet.  The "Whisper detector" — we don't
    // wait for the coordinates to scream before shifting gears.
    float    my_max_plane_kl    = 0.0f;
    // Per-plane max over (Causality, Thermodynamics) — the
    // intent-bearing planes the operator wants us to prioritise.
    float    my_max_caus_therm  = 0.0f;

    if (i < n_clusters && i < SIMT_WIDTH && !i_vetoed) {
        const ContactShellTile* relaxed   = &relaxed_arr[i];
        const ContactShellTile* perturbed = &perturbed_arr[i];

        // Per-plane spectrum pointers — same offset layout in both tiles.
        // Indexed by PLANE_GEO=0 / _CAUS=1 / _THERM=2 / _CHEM=3 from
        // so3_project.cuh.
        const float* p_planes[4] = {
            relaxed->geo_power_spectrum,
            relaxed->caus_power_spectrum,
            relaxed->therm_power_spectrum,
            relaxed->chem_power_spectrum,
        };
        const float* q_planes[4] = {
            perturbed->geo_power_spectrum,
            perturbed->caus_power_spectrum,
            perturbed->therm_power_spectrum,
            perturbed->chem_power_spectrum,
        };

        #pragma unroll
        for (int plane = 0; plane < 4; ++plane) {
            float plane_kl = 0.0f;
            #pragma unroll
            for (int l = 0; l < 6; ++l) {
                float p_raw = p_planes[plane][l];
                float q_raw = q_planes[plane][l];

                const bool p_bad = !isfinite(p_raw);
                const bool q_bad = !isfinite(q_raw);

                // T13 §4.1 hard-trap: NaN/Inf on Causality or
                // Thermodynamics planes is non-physical (those planes
                // carry intent-bearing signal); raise violation.
                if ((p_bad || q_bad) && (plane == 1 || plane == 2)) {
                    my_violation = 1u;
                }

                // Substitute clean placeholders so the running sum stays
                // finite even when Geometry / Chemistry carry NaN drift.
                if (p_bad) p_raw = EPSILON;
                if (q_bad) q_raw = EPSILON;

                const float p     = fmaxf(p_raw, EPSILON);
                const float q     = fmaxf(q_raw, EPSILON);
                const float ratio = p / q;
                // PTX-guarded log2 (defense-in-depth).
                const float log_ratio = prism_log2_with_guard(ratio);

                plane_kl += p * log_ratio;
            }
            my_kl += WEIGHTS[plane] * plane_kl;

            // B.3.2 TIY — track the unweighted per-plane max for the
            // Whisper detector below.  fabsf because plane_kl can go
            // slightly negative on near-equal distributions (the
            // log_ratio guard does NOT enforce non-negativity).
            const float abs_plane_kl = fabsf(plane_kl);
            my_max_plane_kl = fmaxf(my_max_plane_kl, abs_plane_kl);
            // Causality (plane==1) + Thermodynamics (plane==2): the
            // intent-bearing channels.  Their independent threshold
            // is half the global one (operator: "Whisper of a Causal
            // Lead, not the Scream of a Spatial Collision").
            if (plane == 1 || plane == 2) {
                my_max_caus_therm = fmaxf(my_max_caus_therm, abs_plane_kl);
            }
        }
    }

    // ─── Block reduction (sum + OR + max-per-plane) ────────────────
    // B.3.2 TIY — also reduce the max unweighted per-plane KL and the
    // max Causal/Thermo per-plane KL across all clusters so the
    // Whisper detector fires on ANY cluster's intent-bearing spike.
    __shared__ float    s_kl[64];
    __shared__ uint32_t s_violation[64];
    __shared__ float    s_max_plane[64];
    __shared__ float    s_max_caus_therm[64];
    s_kl[i]              = my_kl;
    s_violation[i]       = my_violation;
    s_max_plane[i]       = my_max_plane_kl;
    s_max_caus_therm[i]  = my_max_caus_therm;
    __syncthreads();

    #pragma unroll
    for (uint32_t s = 32u; s > 0u; s >>= 1) {
        if (i < s) {
            s_kl[i]              += s_kl[i + s];
            s_violation[i]       |= s_violation[i + s];
            s_max_plane[i]        = fmaxf(s_max_plane[i],       s_max_plane[i + s]);
            s_max_caus_therm[i]   = fmaxf(s_max_caus_therm[i],  s_max_caus_therm[i + s]);
        }
        __syncthreads();
    }

    if (i == 0u) {
        const float    total_kl              = s_kl[0];
        const uint32_t any_violation         = s_violation[0];
        const float    max_plane_kl_unweighted = s_max_plane[0];
        const float    max_caus_therm_kl       = s_max_caus_therm[0];

        // Race-free write order: divergence first, fence, then SWITCH code.
        adjudicator->current_divergence = total_kl;
        __threadfence();

        uint32_t code;
        if (any_violation != 0u) {
            code = PRISM_ADJ_VIOLATION;
        } else {
            // M1.2.26.RECOUPLE Part 3.3 — KL-Native discovery gate.
            //
            // Pre-RECOUPLE the threshold was computed on the L2-norm
            // SH-coefficient scale (noise_floor_mu[0] ≈ 0.5,
            // noise_floor_sigma[0] ≈ 0.1 → threshold ≈ 1.7) but
            // compared against `total_kl` in log-probability units
            // (range ~1e-5 in production). The M1.2.26 coupling audit
            // proved this scale mismatch made the 12σ gate
            // mathematically unreachable (5+ OOM shortfall regardless
            // of UV dose).
            //
            // Post-RECOUPLE the threshold is dimensionally
            // synchronized: μ/σ are run-locked statistics of
            // `total_kl` itself (Phase 2 of the Split T7 Epoch,
            // steps 51-100, KL(P_frozen || Q_current)). Sentinel
            // total_kl_sigma == 0.0f means those statistics were
            // never written, and the SH-coefficient-scale branch
            // below is the one that adjudicates the run.
            //
            // LIVE PRODUCTION PATH — DO NOT DELETE THE `else` ARM.
            // Earlier revisions of this comment called it the
            // "legacy" threshold.  That word named the SH-coefficient
            // SCALE, not a deprecation; do not read it as one.
            //
            // total_kl_mu / total_kl_sigma have exactly one writer:
            // CapturedAdjudicationPipeline::write_kl_native_mu_sigma
            // in captured_pipeline.rs, which cuMemcpyHtoDAsyncs into
            // offsets 148/152.  Its only live caller chain is the
            // COUPLED lane — run_coupled_twin →
            // execute_strict_coupled_pair →
            // execute_online_split_t7_calibration, all in
            // coupled_md.rs.  The single-topology monomer lane in
            // nhs_rt_full never reaches it, so on that lane both
            // fields keep the 0.0f written by
            // prism_interferometric_adjudicator_zero_kernel in this
            // file, and the SH-coefficient arm is the ONLY arm this
            // gate ever evaluates there.  That lane feeds the arm
            // deliberately: noise_floor_mu/sigma carry the T7-LOCKED
            // priors refined by the cold-hold window, plus
            // PipelineConfig::noise_floor_override (the
            // --noise-floor-mu / --noise-floor-sigma operator flags)
            // when supplied.
            //
            // Delete this arm and the monomer threshold collapses to
            // 0.0f + 12.0f * 0.0f = 0.0f: every frame carrying any
            // nonzero divergence trips CONSTRUCT, PRUNE stops being
            // reachable, and the discovery gate stops discriminating
            // — detection disarmed with no build error and no
            // runtime error to show for it.
            const bool kl_native_ready =
                (adjudicator->total_kl_sigma > 0.0f);
            const float threshold = kl_native_ready
                ? (adjudicator->total_kl_mu
                   + 12.0f * adjudicator->total_kl_sigma)
                : (adjudicator->noise_floor_mu[0]
                   + 12.0f * adjudicator->noise_floor_sigma[0]);

            // ── B.3.2 Total Information Yield (TIY) — operator
            //    addendum 2026-05-02:
            //
            //   "A high-intensity signal in the Causal Plane (Plane 2)
            //    or Thermodynamic Plane (Plane 3) must be able to
            //    trigger Gear 0 (0.5 fs) independently of the
            //    Geometry plane.  The Brain must trigger on the
            //    'Whisper' of a Causal Lead, not the 'Scream' of a
            //    Spatial Collision."
            //
            // Three independent triggers (any → Construct):
            //   1. Weighted-sum trigger:   total_kl > threshold
            //      (geometry-dominant signal — "Scream" detector).
            //   2. Any-plane trigger:      max_plane_kl > threshold
            //      (any single plane spiking — covers Chemistry, etc.).
            //   3. Causal/Thermo trigger:  max_caus_therm > threshold/2
            //      (intent-bearing planes get a 2× sensitivity boost —
            //      the operator-mandated "Whisper" detector).
            // Wave 0 / Task #68 — sign-invariant trigger.  `total_kl` is
            // a 4-plane weighted sum of `Σ p_l · log2(p_l/q_l)`; the
            // log_ratio is signed, so total_kl can land negative when
            // perturbed > relaxed on dominant bands.  Pre-Wave-0 the
            // threshold check was `total_kl > threshold` which silently
            // discarded the entire negative half-plane of real
            // divergence events.  Take fabsf so any spectral departure
            // — in either direction — counts.  `max_plane_kl_unweighted`
            // and `max_caus_therm_kl` already use fabsf at accumulation
            // (lines 264-272), so no change there.
            const bool weighted_trigger = (fabsf(total_kl)         > threshold);
            const bool any_plane_spike  = (max_plane_kl_unweighted > threshold);
            const bool whisper_trigger  = (max_caus_therm_kl       > threshold * 0.5f);
            code = (weighted_trigger || any_plane_spike || whisper_trigger)
                       ? PRISM_ADJ_CONSTRUCT
                       : PRISM_ADJ_PRUNE;
        }

        // G28 SISR campaign-level prune (any vetoed cluster ⇒ frame
        // is suspect; preserve existing semantics from pre-SIMT).
        if (prune_mask != 0ull) {
            code = PRISM_ADJ_PRUNE;
        }

        // M1.2.20.C-B — PTX Momentum Guard override.  The gasp kernel
        // accumulates Σ m_i · Δr_i into com_shift_dev; the
        // momentum_guard_check kernel sets adj.momentum_violation_flag
        // = 1 when the magnitude exceeds 1e-4 Å (operator §3 — a
        // legitimate gasp is an *expansion*, not a translation; if the
        // protein "walks" during the perturbation the SO(3) power
        // spectrum becomes ungrounded noise).  This override fires
        // AFTER the SISR prune so a vetoed-cluster frame whose gasp
        // also drifted bodily still surfaces as VIOLATION (the more
        // diagnostic signal for offline triage).
        // M1.2.20.C-G / T24 — stamp self-auditing reason bits as we
        // make the routing decision.  These persist on the FFI struct
        // across the campaign so the teardown forensic readback shows
        // *why* the adjudicator picked the code it did (operator §4
        // "self-auditing summary").

        if (adjudicator->momentum_violation_flag != 0u) {
            code = PRISM_ADJ_VIOLATION;
            atomicOr(&adjudicator->adjudication_reason_flags, 0x2u);  // MOMENTUM_VIOLATION
        }

        if (any_violation != 0u) {
            // any_violation was set earlier when a NaN/Inf landed on
            // the Causality (plane=1) or Thermodynamics (plane=2) lanes.
            atomicOr(&adjudicator->adjudication_reason_flags, 0x1u);  // NaN_POTENTIAL
        }

        if (prune_mask != 0ull) {
            atomicOr(&adjudicator->adjudication_reason_flags, 0x4u);  // SYMMETRY_VETO
        }

        // LQI Bit-31 Quarantine override (carried over from M1.2.20.C-C).
        // Set by Dynamic T7 reduce when σ² <= 0; read here as the same
        // bit on the renamed adjudication_reason_flags field.
        if ((adjudicator->adjudication_reason_flags & 0x80000000u) != 0u) {
            code = PRISM_ADJ_VIOLATION;
        }

        adjudicator->adjudication_code = code;

        // ─── Anti-Greenfield § 6.2 — legacy_centroid_fallback ──
        // AABB midpoint of cluster 0's relaxed manifold. The legacy
        // single-tile observer cannot represent multi-cluster centroids;
        // cluster 0 is a stable representative for nhs_rt_full's
        // 3-float ABI.
        const ContactShellTile* relaxed_0 = &relaxed_arr[0];
        adjudicator->legacy_centroid_fallback[0] =
            0.5f * (relaxed_0->aabb_min[0] + relaxed_0->aabb_max[0]);
        adjudicator->legacy_centroid_fallback[1] =
            0.5f * (relaxed_0->aabb_min[1] + relaxed_0->aabb_max[1]);
        adjudicator->legacy_centroid_fallback[2] =
            0.5f * (relaxed_0->aabb_min[2] + relaxed_0->aabb_max[2]);
    }
}

/// Zero-fill kernel — initialises a freshly F2-pool-allocated
/// InterferometricAdjudicatorFfi to all-zero state. Called from
/// `prism_interferometric_adjudicator_create` after the pool alloc
/// returns. <<<1, 1, 0, stream>>> single-thread launch.
__global__ void prism_interferometric_adjudicator_zero_kernel(
    InterferometricAdjudicatorFfi* __restrict__ adj
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        adj->noise_floor_mu[i] = 0.0f;
        adj->noise_floor_sigma[i] = 0.0f;
    }
    adj->current_divergence = 0.0f;
    adj->adjudication_code = PRISM_ADJ_PRUNE;
    adj->relaxed_manifold_ptr = nullptr;
    adj->perturbed_manifold_ptr = nullptr;
    adj->start_clock = 0u;
    adj->stop_clock = 0u;
    adj->legacy_centroid_fallback[0] = 0.0f;
    adj->legacy_centroid_fallback[1] = 0.0f;
    adj->legacy_centroid_fallback[2] = 0.0f;
    // gear_override = 0xFF (Auto sentinel) so the gearbox SFA decides.
    // Offset 100 (B.3.2 home, restored Emergency Rectification 2026-05-02).
    // Host writes 0..3 via cuMemcpyHtoDAsync to (adj_dev + 100) to force.
    adj->gear_override = 0xFFu;
    adj->force_prune_mask_ptr = nullptr;
    // potential_energy is an f64 value (not a pointer); 0.0 means
    // "PE not yet computed" — the SFA stability fuse skips the drift
    // check on the first frame to avoid a bogus trap.
    adj->potential_energy = 0.0;
    // d_dt: null until host wires &d_protocol->dt pre-capture via
    // cuMemcpyHtoD into offset 120.
    adj->d_dt = nullptr;
    // d_external_work is a *mut f64 POINTER.  The host wires the
    // F2-pool buffer address into offset 128 pre-capture via cuMemcpyHtoD;
    // Zero-Host Guard §3 rejects null at capture time.  The captured
    // graph emits cuMemsetD8Async at the head of every replay so
    // *d_external_work is fresh at chunk start.
    adj->d_external_work = nullptr;
    // M1.2.20.C-A — Gradient Gasp handles default to neutral state.
    // η_base = 1.0 per Ruling 2; force_burst_step = u32_MAX disables
    // the 10× amplification.  Host overwrites both pre-capture via
    // cuMemcpyHtoDAsync at offsets 136 and 140 respectively.
    adj->gasp_gain_eta = 1.0f;
    adj->force_burst_step = 0xFFFFFFFFu;
    // M1.2.20.C-B — Momentum Guard flag clear (no violation observed).
    // The captured graph emits a cuMemsetD8Async on this 4-byte field
    // at the head of every replay window so each chunk starts with a
    // clean violation state.
    adj->momentum_violation_flag = 0u;
    // M1.2.26.RECOUPLE Part 2 — KL-native μ/σ start at zero. Phase 2
    // of the Split T7 Epoch (steps 51-100) writes these. Until then
    // F1 SWITCH falls back to legacy noise_floor_mu[0] threshold.
    adj->total_kl_mu    = 0.0f;
    adj->total_kl_sigma = 0.0f;
    // M1.2.20.C-G / T24 — adjudication_reason_flags cleared. Field
    // is NOT reset per-replay (it accumulates reasons across the
    // campaign for the teardown forensic readback); the zero kernel
    // runs once at engine init.
    adj->adjudication_reason_flags = 0u;
    #pragma unroll
    for (int k = 0; k < 96; ++k) {
        adj->_reserved_m1_2_26[k] = 0u;
    }
}

/// Noise-floor update kernel — Welford-style running mean+stddev over
/// the latest "Cool" frame's 6-band power spectrum. Single-thread.
__global__ void prism_interferometric_adjudicator_noise_kernel(
    InterferometricAdjudicatorFfi* __restrict__ adj,
    const float* __restrict__ cool_power_spectrum
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // First-pass running update: μ_new = 0.5·μ_old + 0.5·sample.
    // This is an EMA, not strict Welford, to keep the noise-floor
    // adaptive on a per-frame timescale. σ tracks abs-deviation as a
    // rough proxy for stddev — a future enhancement is the proper
    // Welford recurrence M2 += (sample−μ_old)·(sample−μ_new).
    #pragma unroll
    for (int l = 0; l < 6; ++l) {
        const float sample = cool_power_spectrum[l];
        if (!isfinite(sample)) continue;  // skip dirty samples
        const float mu_old = adj->noise_floor_mu[l];
        const float mu_new = 0.5f * mu_old + 0.5f * sample;
        const float dev    = fabsf(sample - mu_new);
        const float sig_old = adj->noise_floor_sigma[l];
        const float sig_new = 0.5f * sig_old + 0.5f * dev;
        adj->noise_floor_mu[l] = mu_new;
        adj->noise_floor_sigma[l] = sig_new;
    }
}

/// Link-probe kernel — returns a sentinel via a stream-ordered single
/// write. Used by the Rust-side test to detect FFI / build-system
/// drift.
__global__ void prism_interferometric_adjudicator_probe_kernel(uint32_t* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *out = 0xAD31u;
}

// ════════════════════════════════════════════════════════════════════
// T3 — ASC Boundary Repulsion Tensor kernel
// ════════════════════════════════════════════════════════════════════

/// Per-atom outward-repulsion kernel. Reads the relaxed manifold's
/// AABB, computes the centroid X_c, and adds an outward force
/// F_i = α · Δ_AB · (x_i − X_c) into the existing fused_engine
/// d_forces buffer. Anti-Greenfield § 2.1 surgical extension.
///
/// Pointer-stability invariant: every device pointer read here MUST
/// be allocated outside the captured-graph WHILE region. The F2
/// pool's UINT64_MAX release threshold guarantees that.
///
/// Branchless cluster-membership: `d_atom_in_cluster[i]` is `0` or `1`;
/// the updates are gated by an integer multiply against that mask
/// rather than an early-return branch, eliminating warp divergence
/// when a warp straddles the cluster boundary.
__global__ void prism_asc_apply_kernel(
    const InterferometricAdjudicatorFfi* __restrict__ adj,
    float*          __restrict__ d_forces,
    const float*    __restrict__ d_atom_positions,
    const uint32_t* __restrict__ d_atom_in_cluster,
    int32_t         n_atoms,
    float           steering_gain_alpha,
    // M1.2.18-P3.2 — per-atom PE accumulator.  When non-null, the
    // ASC kernel folds V_ASC = -½·α·Δ_AB·‖x_i − X_c‖²·mask into
    // pe_components[i] so the SFA's drift fuse sees the steering
    // potential as part of V_t (operator §3.2: "the SFA fuse should
    // not fire on Intentional Perturbation").
    double*         __restrict__ pe_components,
    // P7a — ASC APPLICATION EVIDENCE (both nullable).
    //
    // Whether ASC actually perturbs the trajectory was previously only
    // INFERABLE: the kernel self-gates on adjudication_code, the host
    // gates on cfg.asc.is_some(), and the buffer it accumulates into is
    // rewritten by `forces[tid] = local_force` in the fused engine's
    // FORCE_N / FORCE_NP1 phases. Nothing MEASURED whether a
    // contribution was made, so no receipt could state it and no
    // artifact could be trusted to reflect it.
    //
    //   d_asc_applied_atoms : atoms that received a non-zero steering
    //                         force this launch. Masked-out atoms carry
    //                         scale == 0 and are NOT counted.
    //   d_asc_applied_sum_f : sum of |F_ASC| over those atoms, so the
    //                         receipt reports magnitude, not just
    //                         incidence.
    //
    // Accumulated with ONE global atomicAdd per block after a shared
    // reduction. Callers passing nullptr for both pay no reduction cost.
    uint32_t*       __restrict__ d_asc_applied_atoms,
    float*          __restrict__ d_asc_applied_sum_f
) {
    // P7a — block-reduction scratch. Declared before any branch, and the
    // whole body is structured as `if (active) { ... }` rather than early
    // returns, because a `return` taken by only part of the block would
    // desynchronise the __syncthreads() below. The final block is
    // partially populated whenever n_atoms is not a multiple of blockDim,
    // so that case is the normal one, not an edge case.
    __shared__ unsigned int s_applied_n;
    __shared__ float        s_applied_f;
    const bool want_evidence =
        (d_asc_applied_atoms != nullptr) || (d_asc_applied_sum_f != nullptr);

    if (want_evidence) {
        if (threadIdx.x == 0) {
            s_applied_n = 0u;
            s_applied_f = 0.0f;
        }
        __syncthreads();
    }

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Defensive: only fire on Construct (code 1). The F1 SWITCH already
    // routed us here only when code == 1, but read-back guards against
    // pointer-stability bugs upstream.
    const uint32_t code = adj->adjudication_code;
    const ContactShellTile* tile = adj->relaxed_manifold_ptr;
    const bool active =
        (i < n_atoms) && (code == PRISM_ADJ_CONSTRUCT) && (tile != nullptr);

    if (active) {
        const float xc0 = 0.5f * (tile->aabb_min[0] + tile->aabb_max[0]);
        const float xc1 = 0.5f * (tile->aabb_min[1] + tile->aabb_max[1]);
        const float xc2 = 0.5f * (tile->aabb_min[2] + tile->aabb_max[2]);

        const float xi0 = d_atom_positions[i * 3 + 0];
        const float xi1 = d_atom_positions[i * 3 + 1];
        const float xi2 = d_atom_positions[i * 3 + 2];

        const float dx = xi0 - xc0;
        const float dy = xi1 - xc1;
        const float dz = xi2 - xc2;

        // Branchless mask multiply — keeps every lane in lockstep.
        const float mask  = static_cast<float>(d_atom_in_cluster[i]);
        const float scale = steering_gain_alpha * adj->current_divergence * mask;

        const float fx = scale * dx;
        const float fy = scale * dy;
        const float fz = scale * dz;

        d_forces[i * 3 + 0] += fx;
        d_forces[i * 3 + 1] += fy;
        d_forces[i * 3 + 2] += fz;

        // ── M1.2.18-P3.2 — ASC steering potential accumulation ──
        //
        //   F_ASC = α · Δ_AB · (x_i − X_c) · mask          (anti-restraint)
        //   V_ASC = −½ · α · Δ_AB · ‖x_i − X_c‖² · mask    (∫ −F·dr)
        //
        // The minus sign is intentional: ASC PUSHES atoms outward
        // (positive F along the radial direction), so its potential is
        // INVERTED ("anti-restraint", confining well becomes a hill).
        // Folding V_ASC into pe_components means V_t already reflects the
        // steering work; the SFA drift fuse can't false-positive on
        // Construct frames just because we're injecting steering force.
        if (pe_components != nullptr) {
            const float r2 = dx * dx + dy * dy + dz * dz;
            const double v_asc = -0.5 * (double)scale * (double)r2;
            pe_components[i] += v_asc;
        }

        // P7a — record that this atom actually received steering.
        if (want_evidence) {
            const float fmag = sqrtf(fx * fx + fy * fy + fz * fz);
            if (fmag > 0.0f) {
                atomicAdd(&s_applied_n, 1u);
                atomicAdd(&s_applied_f, fmag);
            }
        }
    }

    // P7a — one global atomicAdd per block. Every thread in the block
    // reaches this barrier, active or not.
    if (want_evidence) {
        __syncthreads();
        if (threadIdx.x == 0) {
            if (d_asc_applied_atoms != nullptr && s_applied_n != 0u)
                atomicAdd(d_asc_applied_atoms, s_applied_n);
            if (d_asc_applied_sum_f != nullptr && s_applied_f != 0.0f)
                atomicAdd(d_asc_applied_sum_f, s_applied_f);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// T4 — clock64 pipeline timing bookends (PTX-direct)
// ════════════════════════════════════════════════════════════════════

/// Returns the SM cycle counter via the canonical `mov.u64 %0, %clock64`
/// PTX instruction. One PTX instruction; <1ns overhead. Equivalent to
/// the CUDA `clock64()` intrinsic but emitted as inline PTX per the
/// operator's directive § 3.1.
__device__ __forceinline__ uint64_t prism_clock64_ptx() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
    return t;
}

__global__ void prism_pipeline_clock_start_kernel(
    InterferometricAdjudicatorFfi* __restrict__ adj
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    adj->start_clock = prism_clock64_ptx();
}

__global__ void prism_pipeline_clock_stop_kernel(
    InterferometricAdjudicatorFfi* __restrict__ adj
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    adj->stop_clock = prism_clock64_ptx();
}

// ════════════════════════════════════════════════════════════════════
// Host-callable launchers (extern "C") — match Rust FFI declarations
// in interferometric_adjudicator.rs::ffi.
// ════════════════════════════════════════════════════════════════════

extern "C" {

uint32_t prism_interferometric_adjudicator_link_probe(void) {
    // Allocate a 4-byte pinned host buffer, run the probe, sync, return.
    // This pattern mirrors vram_pool's link probe (which uses a static
    // device buffer); we use a heap one-shot here since the probe is
    // one-time-only at startup.
    uint32_t* d_out = nullptr;
    cudaError_t err = cudaMalloc(&d_out, sizeof(uint32_t));
    if (err != cudaSuccess) return 0xDEADu;
    prism_interferometric_adjudicator_probe_kernel<<<1, 1, 0, 0>>>(d_out);
    cudaDeviceSynchronize();
    uint32_t value = 0;
    err = cudaMemcpy(&value, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    return (err == cudaSuccess) ? value : 0xDEADu;
}

int prism_interferometric_adjudicator_init_constants(void* stream_v) {
    // The __constant__ symbol c_mu_01_sq is statically initialised at
    // module load (see file-scope init above). No host-side memcpy is
    // required; this function exists for explicit ABI symmetry with
    // future LUTs that need runtime calibration.
    //
    // Returns cudaSuccess unconditionally — callers can rely on this
    // being a stable no-op while the LUT is compile-time-constant.
    (void)stream_v;
    return static_cast<int>(cudaSuccess);
}

int prism_interferometric_adjudicator_create(
    void* pool_v, void* stream_v,
    InterferometricAdjudicatorFfi** out_ptr
) {
    // F2 pool allocation via the existing vram_pool wrapper. We do
    // NOT cudaMalloc — that's host-syncing and breaks captured-graph
    // replay (Anti-Greenfield § 3.2 forbids it for live state).
    extern int prism_vram_pool_alloc_async(void*, uint64_t, void*, void**);
    void* raw = nullptr;
    int rc = prism_vram_pool_alloc_async(
        pool_v,
        static_cast<uint64_t>(sizeof(InterferometricAdjudicatorFfi)),
        stream_v, &raw);
    if (rc != static_cast<int>(cudaSuccess)) return rc;
    if (raw == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);

    // Zero-init on the same stream — stream-ordered ensures the alloc
    // is visible before the kernel's writes.
    auto* adj = static_cast<InterferometricAdjudicatorFfi*>(raw);
    cudaStream_t s = static_cast<cudaStream_t>(stream_v);
    prism_interferometric_adjudicator_zero_kernel<<<1, 1, 0, s>>>(adj);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int>(err);

    *out_ptr = adj;
    return static_cast<int>(cudaSuccess);
}

int prism_interferometric_adjudicator_destroy(
    InterferometricAdjudicatorFfi* adj, void* stream_v
) {
    extern int prism_vram_pool_free_async(void*, void*);
    return prism_vram_pool_free_async(static_cast<void*>(adj), stream_v);
}

int prism_interferometric_adjudicator_step(
    InterferometricAdjudicatorFfi* adj,
    uint32_t                       n_clusters,
    void*                          stream_v
) {
    // T13 — SIMT 4-plane KL fusion. Single block of 64 threads; each
    // thread processes one cluster (capped at 64 clusters per launch;
    // multi-block adjudication deferred to a separate refactor).
    cudaStream_t s = static_cast<cudaStream_t>(stream_v);
    prism_interferometric_adjudicator_step_kernel<<<1, 64, 0, s>>>(adj, n_clusters);
    return static_cast<int>(cudaGetLastError());
}

int prism_interferometric_adjudicator_update_noise_floor(
    InterferometricAdjudicatorFfi* adj,
    const float* cool_power_spectrum,
    void* stream_v
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream_v);
    prism_interferometric_adjudicator_noise_kernel<<<1, 1, 0, s>>>(
        adj, cool_power_spectrum);
    return static_cast<int>(cudaGetLastError());
}

// ────────────────────────────────────────────────────────────────────
// T3 / T4 host launchers
// ────────────────────────────────────────────────────────────────────

int prism_asc_apply(
    const InterferometricAdjudicatorFfi* adj,
    float* d_forces,
    const float* d_atom_positions,
    const uint32_t* d_atom_in_cluster,
    int32_t n_atoms,
    float steering_gain_alpha,
    void* stream_v,
    // M1.2.18-P3.2 — per-atom PE accumulator (nullable).  When non-
    // null, the ASC kernel folds V_ASC into pe_components[i] so the
    // SFA's drift fuse sees the steering potential as part of V_t.
    double* pe_components,
    // P7a — nullable ASC application evidence counters.
    uint32_t* d_asc_applied_atoms,
    float*    d_asc_applied_sum_f
) {
    if (n_atoms <= 0) return static_cast<int>(cudaSuccess);
    cudaStream_t s = static_cast<cudaStream_t>(stream_v);
    constexpr int BLOCK = 256;
    const int grid = (n_atoms + BLOCK - 1) / BLOCK;
    prism_asc_apply_kernel<<<grid, BLOCK, 0, s>>>(
        adj, d_forces, d_atom_positions, d_atom_in_cluster,
        n_atoms, steering_gain_alpha, pe_components,
        d_asc_applied_atoms, d_asc_applied_sum_f);
    return static_cast<int>(cudaGetLastError());
}

int prism_pipeline_clock_start(
    InterferometricAdjudicatorFfi* adj, void* stream_v
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream_v);
    prism_pipeline_clock_start_kernel<<<1, 1, 0, s>>>(adj);
    return static_cast<int>(cudaGetLastError());
}

int prism_pipeline_clock_stop(
    InterferometricAdjudicatorFfi* adj, void* stream_v
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream_v);
    prism_pipeline_clock_stop_kernel<<<1, 1, 0, s>>>(adj);
    return static_cast<int>(cudaGetLastError());
}

// ────────────────────────────────────────────────────────────────────
// DAG-COND-WIRE host launchers
// ────────────────────────────────────────────────────────────────────

const uint32_t* prism_get_adjudication_code_devptr(
    const InterferometricAdjudicatorFfi* adj
) {
    // Field is at byte offset 52 within the 128-aligned struct.
    // Pointer is stable for the campaign per F2 pool's
    // ReleaseThreshold = UINT64_MAX guarantee. Always 4-byte aligned;
    // never crosses a 128-byte L1 sector boundary (the entire
    // 128-byte struct fits in one sector).
    return &adj->adjudication_code;
}

}  // extern "C"

// ────────────────────────────────────────────────────────────────────
// F1 SWITCH translation kernel (CUDA 12.4+: cudaGraphSetConditional)
// ────────────────────────────────────────────────────────────────────
//
// Conditional-node support requires CUDA 12.4+. Older toolkits define
// neither `cudaGraphConditionalHandle` nor `cudaGraphSetConditional`,
// so we feature-gate via the toolkit version macro to keep the
// archive linkable on transitional dev environments.
//
// `cudaGraphSetConditional` is documented in:
//   CUDA Programming Guide § 3.2.8.7.4 "Conditional Node Setting"
// Available since CUDA 12.4 (toolkit version 12040). Project ships on
// CUDA 13.x (per build.rs and the e9628229 F2-pool commit), so this
// path is always live in production builds.

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040

__global__ void prism_adj_set_conditional_kernel(
    cudaGraphConditionalHandle handle,
    const InterferometricAdjudicatorFfi* __restrict__ adj
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // Read the memory-resident code written by T2's adjudicator
    // kernel. The captured-graph dependency edge from T2 → here
    // guarantees stream-order visibility (operator mandate § 2.3).
    const uint32_t code = adj->adjudication_code;
    // Forward to the F1 SWITCH handle. The 3 valid values
    // (PRUNE=0, CONSTRUCT=1, VIOLATION=2) map to the SWITCH's
    // 3 sub-graph branches.
    cudaGraphSetConditional(handle, code);
}

extern "C" int prism_adj_set_conditional(
    uint64_t handle_v,
    const InterferometricAdjudicatorFfi* adj,
    void* stream_v
) {
    // cudaGraphConditionalHandle is `typedef unsigned long long`
    // (driver_types.h:3229), so the value is a direct copy.
    cudaGraphConditionalHandle handle =
        static_cast<cudaGraphConditionalHandle>(handle_v);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_v);
    prism_adj_set_conditional_kernel<<<1, 1, 0, stream>>>(handle, adj);
    return static_cast<int>(cudaGetLastError());
}

#else  // CUDART_VERSION < 12040

extern "C" int prism_adj_set_conditional(
    uint64_t /*handle_v*/,
    const InterferometricAdjudicatorFfi* /*adj*/,
    void* /*stream_v*/
) {
    // Pre-CUDA-12.4 toolchain: conditional nodes are unavailable.
    // Return cudaErrorNotSupported so callers can fail loudly.
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12040

// ════════════════════════════════════════════════════════════════════
// V2 C-ABI BYPASS — Native F1 SWITCH forge
// ════════════════════════════════════════════════════════════════════
//
// Single-call host-side helper that:
//   1. Creates a cudaGraphConditionalHandle bound to `graph`.
//   2. Populates cudaGraphNodeParams for a 3-way SWITCH conditional node.
//   3. Adds the conditional node to the graph.
//   4. Locks the happens-before edge: adjudicator_node → conditional_node.
//
// Requires CUDA 12.6+ for cudaGraphCondTypeSwitch. Production toolchain
// is CUDA 13.x (CUDART_VERSION = 13020 per system headers), so the
// path below is always live; the gate exists for transitional
// dev-environment safety.

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12060

extern "C" int prism_wire_f1_switch_ffi(
    cudaGraph_t      graph,
    cudaGraphNode_t  adjudicator_node,
    const uint32_t*  predicate_dev_ptr,
    cudaGraphNode_t* out_conditional_node
) {
    if (graph == nullptr || out_conditional_node == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // Step 1: Create the F1 SWITCH handle bound to this graph.
    // defaultLaunchValue = 0 ⇒ Prune by default if the bridge kernel
    // hasn't updated the handle (e.g., first frame before any
    // adjudicator step has run).
    cudaGraphConditionalHandle handle = 0;
    cudaError_t err = cudaGraphConditionalHandleCreate(
        &handle, graph,
        /*defaultLaunchValue=*/ 0u,
        /*flags=*/              cudaGraphCondAssignDefault);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    // `predicate_dev_ptr` is wiring metadata — it is the device
    // address of `adj->adjudication_code` (offset 52 inside the
    // 128-aligned InterferometricAdjudicatorFfi). The runtime
    // conditional-handle API is handle-based, not pointer-based, so
    // the pointer is not consumed directly here; the downstream
    // `prism_adj_set_conditional_kernel` reads from it to forward
    // the value into the handle via `cudaGraphSetConditional`.
    // Retained in the function signature for wiring-contract clarity
    // and forward-compatibility with a possible future device-
    // pointer-conditional API.
    (void)predicate_dev_ptr;

    // Step 2: Populate cudaGraphNodeParams for the conditional node.
    // The 3-way SWITCH maps directly to the F1 routing table:
    //   Case 0 → Prune sub-graph
    //   Case 1 → Construct (bisimulation + ASC) sub-graph
    //   Case 2 → Violation (PTX trap) sub-graph
    cudaGraphNodeParams nodeParams{};
    nodeParams.type                 = cudaGraphNodeTypeConditional;
    nodeParams.conditional.handle   = handle;
    nodeParams.conditional.type     = cudaGraphCondTypeSwitch;
    nodeParams.conditional.size     = 3u;
    // ctx = NULL ⇒ runtime fills in from the current device context
    // at node-creation time. Avoids dragging the driver-API CUcontext
    // into this TU.
    nodeParams.conditional.ctx      = nullptr;

    // Step 3: Add the conditional node. Pass NULL for both
    // pDependencies and dependencyData (CUDA 13.x signature) — we
    // install the dependency edge in step 4 to make the wiring
    // contract explicit + match the operator's numbered ordering.
    //
    // CUDA 13.x cudaGraphAddNode signature:
    //   cudaError_t cudaGraphAddNode(
    //       cudaGraphNode_t* pGraphNode,
    //       cudaGraph_t graph,
    //       const cudaGraphNode_t* pDependencies,
    //       const cudaGraphEdgeData* dependencyData,   ← NEW in 13.x
    //       size_t numDependencies,
    //       cudaGraphNodeParams* nodeParams
    //   );
    err = cudaGraphAddNode(
        out_conditional_node, graph,
        /*pDependencies=*/   nullptr,
        /*dependencyData=*/  nullptr,
        /*numDependencies=*/ 0u,
        &nodeParams);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    // Step 4: Gate G19 — happens-before edge.
    // CUDA 13.x cudaGraphAddDependencies signature:
    //   cudaError_t cudaGraphAddDependencies(
    //       cudaGraph_t graph,
    //       const cudaGraphNode_t* from,
    //       const cudaGraphNode_t* to,
    //       const cudaGraphEdgeData* edgeData,         ← NEW in 13.x
    //       size_t numDependencies
    //   );
    // Both from/to are 1-element arrays via &adjudicator_node and
    // out_conditional_node respectively. edgeData = nullptr selects
    // the default cudaGraphEdgeTypeProgram + cudaGraphFullPort
    // edge attributes (the ordinary "happens-before" edge).
    err = cudaGraphAddDependencies(
        graph,
        /*from=*/             &adjudicator_node,
        /*to=*/               out_conditional_node,
        /*edgeData=*/         nullptr,
        /*numDependencies=*/  1u);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    return static_cast<int>(cudaSuccess);
}

#else  // CUDART_VERSION < 12060

extern "C" int prism_wire_f1_switch_ffi(
    cudaGraph_t      /*graph*/,
    cudaGraphNode_t  /*adjudicator_node*/,
    const uint32_t*  /*predicate_dev_ptr*/,
    cudaGraphNode_t* /*out_conditional_node*/
) {
    // Pre-CUDA-12.6: cudaGraphCondTypeSwitch (3-way SWITCH) is not
    // available. Caller should fall back to a chain of If-conditional
    // nodes if 2-way branching is acceptable, or upgrade the toolkit.
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12060

// ════════════════════════════════════════════════════════════════════
// T12 Pre-Flight — G26 Chronometric Gearbox 4-way SWITCH forge
// ════════════════════════════════════════════════════════════════════
//
// Sibling of `prism_wire_f1_switch_ffi` above, but with `size = 4`
// and returns ALL FOUR body-subgraph handles via `out_body_subgraphs`
// so the caller can populate them with PointerSwap + VelocityRescale
// kernels (Wave B). Wave A only INSTANTIATES the SWITCH skeleton —
// the body sub-graphs are returned to Rust unpopulated and the SWITCH
// fires Gear 0 (default) for every frame.
//
// Predicate forwarding: caller wires a separate bridge kernel that
// reads the gear_id source (TBD in Wave B — likely `adj->gear_id`
// once the integrator refactor lands) and calls
// `cudaGraphSetConditional(handle, gear_id)` to route to the matching
// sub-graph.  This Pre-Flight helper does NOT capture the bridge —
// it only creates the conditional node and exposes the sub-graphs.

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12060

extern "C" int prism_wire_g26_gearbox_ffi(
    cudaGraph_t      graph,
    cudaGraphNode_t  predicate_node,
    const uint32_t*  predicate_dev_ptr,
    cudaGraphNode_t* out_conditional_node,
    cudaGraph_t*     out_body_subgraphs    /* [4] */
) {
    if (graph == nullptr ||
        out_conditional_node == nullptr ||
        out_body_subgraphs == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // Initialise body-subgraph slots to null so a partial failure
    // never leaves stale handles behind for the caller to misuse.
    for (int i = 0; i < 4; ++i) out_body_subgraphs[i] = nullptr;

    // Step 1 — bind a fresh conditional handle to this graph. Default
    // launch value 0 ⇒ Gear 0 (high-resolution 0.5 fs) is the safe
    // fallback if the predicate forwarder hasn't fired yet.
    cudaGraphConditionalHandle handle = 0;
    cudaError_t err = cudaGraphConditionalHandleCreate(
        &handle, graph,
        /*defaultLaunchValue=*/ 0u,
        /*flags=*/              cudaGraphCondAssignDefault);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    // The predicate device pointer is wiring metadata for the future
    // bridge kernel; the runtime API itself is handle-based, not
    // pointer-based, so we do not consume it here. Suppress unused-
    // parameter warning while keeping the signature symmetric with
    // `prism_wire_f1_switch_ffi`.
    (void)predicate_dev_ptr;

    // Step 2 — populate cudaGraphNodeParams for the 4-way SWITCH.
    //   Case 0 → Gear 0 (0.5 fs, high-resolution capture)
    //   Case 1 → Gear 1 (2.0 fs, default monitoring)
    //   Case 2 → Gear 2 (4.0 fs, HMR-stabilised sprint)
    //   Case 3 → Gear 3 (abort sub-graph; PTX trap kernel target)
    cudaGraphNodeParams nodeParams{};
    nodeParams.type                 = cudaGraphNodeTypeConditional;
    nodeParams.conditional.handle   = handle;
    nodeParams.conditional.type     = cudaGraphCondTypeSwitch;
    nodeParams.conditional.size     = 4u;
    nodeParams.conditional.ctx      = nullptr;

    // Step 3 — add the conditional node, depending on the predicate
    // node so the SWITCH only fires after the upstream gear_id write
    // is L2-visible.
    err = cudaGraphAddNode(
        out_conditional_node, graph,
        /*pDependencies=*/   &predicate_node,
        /*dependencyData=*/  nullptr,
        /*numDependencies=*/ 1u,
        &nodeParams);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    // Step 4 — Runtime API SWITCH/IF nodes write the body-subgraph
    // handles into `nodeParams.conditional.phGraph_out` on creation.
    // The struct field is set by `cudaGraphAddNode` itself; we copy
    // the handles to the caller's output array.  Note: on CUDA 13.x
    // the runtime fills phGraph_out for IF-type nodes via cudarc's
    // driver bridge; for SWITCH-type, the same call is the documented
    // path (CUDA Programming Guide § 3.2.8.7.4). If the toolkit ever
    // diverges we fall back to a follow-up cudaGraphConditionalHandleCreate
    // path — out-of-scope for Wave A.
    for (int i = 0; i < 4; ++i) {
        out_body_subgraphs[i] = nodeParams.conditional.phGraph_out
                                ? nodeParams.conditional.phGraph_out[i]
                                : nullptr;
    }

    return static_cast<int>(cudaSuccess);
}

#else  // CUDART_VERSION < 12060

extern "C" int prism_wire_g26_gearbox_ffi(
    cudaGraph_t      /*graph*/,
    cudaGraphNode_t  /*predicate_node*/,
    const uint32_t*  /*predicate_dev_ptr*/,
    cudaGraphNode_t* /*out_conditional_node*/,
    cudaGraph_t*     /*out_body_subgraphs*/
) {
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12060

// ════════════════════════════════════════════════════════════════════
// B.3.2-FULL — Conditional handle creation + SWITCH wire with
// pre-existing handle (for the captured-graph integration path).
// ════════════════════════════════════════════════════════════════════
//
// During captured-graph build:
//   1. Caller is mid-capture; calls cuStreamGetCaptureInfo to get the
//      in-progress graph handle.
//   2. Calls prism_gearbox_create_handle_ffi(in_progress_graph, &handle).
//   3. Launches the predicate-bridge kernel WITH `handle` as its arg —
//      the kernel-node captured into the graph will use this exact
//      handle value at every launch's cudaGraphSetConditional call.
//   4. Continues capture; calls cuStreamEndCapture.
//   5. Post-capture: prism_gearbox_wire_with_handle_ffi(graph, bridge_node,
//      handle, &cond_node, body_subgraphs) — adds the SWITCH conditional
//      referencing the same handle downstream of bridge_node.
//   6. prism_gearbox_populate_switch_bodies_ffi populates the bodies.
//
// This is the operator-mandated B.3.2-FULL path: handle is created
// during capture so the bridge kernel and SWITCH share it; the SWITCH
// is added post-capture (since cuGraphConditionalNode cannot be
// directly captured as a kernel node).

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12060

extern "C" int prism_gearbox_create_handle_ffi(
    cudaGraph_t   graph,
    uint32_t      default_value,
    uint64_t*     out_handle
) {
    if (graph == nullptr || out_handle == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaGraphConditionalHandle handle = 0;
    cudaError_t err = cudaGraphConditionalHandleCreate(
        &handle, graph,
        /*defaultLaunchValue=*/ default_value,
        /*flags=*/              cudaGraphCondAssignDefault);
    if (err != cudaSuccess) return static_cast<int>(err);
    *out_handle = static_cast<uint64_t>(handle);
    return static_cast<int>(cudaSuccess);
}

extern "C" int prism_gearbox_wire_with_handle_ffi(
    cudaGraph_t      graph,
    cudaGraphNode_t  predicate_node,
    uint64_t         handle_v,
    cudaGraphNode_t* out_conditional_node,
    cudaGraph_t*     out_body_subgraphs    /* [4] */
) {
    if (graph == nullptr ||
        out_conditional_node == nullptr ||
        out_body_subgraphs == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaGraphConditionalHandle handle =
        static_cast<cudaGraphConditionalHandle>(handle_v);

    for (int i = 0; i < 4; ++i) out_body_subgraphs[i] = nullptr;

    cudaGraphNodeParams nodeParams{};
    nodeParams.type                 = cudaGraphNodeTypeConditional;
    nodeParams.conditional.handle   = handle;
    nodeParams.conditional.type     = cudaGraphCondTypeSwitch;
    nodeParams.conditional.size     = 4u;
    nodeParams.conditional.ctx      = nullptr;

    // Downstream of predicate_node so the SWITCH only fires after the
    // bridge kernel has called cudaGraphSetConditional.
    cudaError_t err = cudaGraphAddNode(
        out_conditional_node, graph,
        /*pDependencies=*/   &predicate_node,
        /*dependencyData=*/  nullptr,
        /*numDependencies=*/ 1u,
        &nodeParams);
    if (err != cudaSuccess) return static_cast<int>(err);

    // Driver-probe-confirmed: phGraph_out is populated for SWITCH-type
    // on CUDA 13.x.  Copy handles for the caller.
    for (int i = 0; i < 4; ++i) {
        out_body_subgraphs[i] = nodeParams.conditional.phGraph_out
                                ? nodeParams.conditional.phGraph_out[i]
                                : nullptr;
    }
    // Defensive — if any handle is null the populator will fail with
    // its own InvalidValue (the operator-mandated Smoking-Gun signal).

    return static_cast<int>(cudaSuccess);
}

#else

extern "C" int prism_gearbox_create_handle_ffi(
    cudaGraph_t /*graph*/, uint32_t /*dv*/, uint64_t* /*out*/
) {
    return static_cast<int>(cudaErrorNotSupported);
}

extern "C" int prism_gearbox_wire_with_handle_ffi(
    cudaGraph_t /*graph*/,
    cudaGraphNode_t /*predicate_node*/,
    uint64_t /*handle_v*/,
    cudaGraphNode_t* /*out_conditional_node*/,
    cudaGraph_t* /*out_body_subgraphs*/
) {
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12060

// ════════════════════════════════════════════════════════════════════
// F1-PARENT-SWITCH-001 — Parent-owned F1 SWITCH (size=3)
// ════════════════════════════════════════════════════════════════════
//
// Sibling family of the G26 parent-owned SWITCH helpers above; reads
// `adj->adjudication_code` (offset 52) instead of `cruise->current_gear`,
// and SWITCH `size` is hard-coded to 3 (not 4 — F1 has Prune/Construct/
// Violation, no fourth case). Layout mirrors the G26 B.3.2-FULL pattern
// (handle creation + wire-with-handle + body populator + predicate-
// bridge kernel + host shim) so the Rust orchestrator can reuse the
// G26 wire-up shape verbatim, swapping symbol names only.
//
// **Splice-legality invariant (GRAPH-SPLICE-001):**
//   The F1 predicate-bridge kernel is **parent-graph-resident** — it
//   is launched from a kernel-node sitting in the PARENT graph, NOT
//   inside the spliced child template. The legacy in-child variant
//   `prism_wire_f1_switch_ffi` (above, ~line 865) is the
//   GRAPH-SPLICE-001 anti-pattern preserved only for the existing test
//   `monolithic_pipeline_v2_ignition_smoke_with_hook`; the production
//   monolithic-discovery path goes through THIS family.
//
// **Body topology (per ticket §3 / scout §6):**
//   case 0 → Prune     : empty (no-op; default flow continues)
//   case 1 → Construct : empty (no-op; ASC parent-side reserved)
//   case 2 → Violation : PTX `trap;` kernel — short-circuits before G26
//
// Default launch value = 0 (Prune) so the SWITCH routes to the no-op
// body when the bridge kernel has not yet written via
// `cudaGraphSetConditional` (e.g., first frame before adjudicator step).
//
// Requires CUDA 12.6+ for `cudaGraphCondTypeSwitch`. Production
// toolchain is CUDA 13.x; pre-12.6 stubs return cudaErrorNotSupported.

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12060

// ─── F1 violation trap kernel (case 2 / Violation body) ─────────────────
//
// Single-thread PTX `trap;` — halts the GPU stream and surfaces
// CUDA_ERROR_LAUNCH_FAILED on the next host sync. Mirrors
// `prism_gearbox_trap_kernel` shape; F1-private symbol so the legacy
// in-child wire path and the new parent-owned path do not collide.
extern "C"
__global__ void prism_f1_violation_trap_kernel() {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    asm volatile("trap;");
}

// ─── F1 predicate-bridge kernel (PARENT-GRAPH-RESIDENT) ────────────────
//
// Reads `adj->adjudication_code` (offset 52 inside the 128-aligned
// `InterferometricAdjudicatorFfi`) and forwards `code & mask` into the
// SWITCH conditional handle via `cudaGraphSetConditional`. Mirrors
// `prism_gearbox_predicate_bridge_kernel` (gearbox.cu:437) but with no
// gear_override consult — F1's predicate is the raw adjudication code
// produced by the child adjudicator step, no host-side shadow path.
//
// **Mask = 0x3** (2-bit defensive). The 3 valid codes (Prune=0,
// Construct=1, Violation=2) all fit in 2 bits; any code value 3 (which
// is not produced by the kernel) would route to a non-existent body
// and cause a CUDA error, so the mask is a hard guard, not optional.
//
// **Single-thread launch** — `<<<1, 1, 0, stream>>>`. Branchless on the
// happy path; the early-return for non-(0,0) thread is a no-op since
// the launch shape is single-thread by construction.
//
// **Capture window**: this kernel is launched from the PARENT graph,
// **after** the child adjudicator graph node returns. Adding it inside
// the child template is a GRAPH-SPLICE-001 violation (the legacy
// `prism_wire_f1_switch_ffi` path).
extern "C"
__global__ void prism_f1_predicate_bridge_kernel(
    const uint32_t*             __restrict__   d_adjudication_code,
    cudaGraphConditionalHandle                  handle,
    uint32_t                                    mask)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (d_adjudication_code == nullptr) {
        // Defensive: route to the default (Prune) body if the
        // adjudication-code device pointer was not wired. The default
        // launch value of the handle is also 0, so this is a true no-op
        // — but writing it explicitly makes the post-condition certain.
        cudaGraphSetConditional(handle, 0u);
        return;
    }
    const uint32_t code = (*d_adjudication_code) & mask;
    cudaGraphSetConditional(handle, code);
}

// ─── F1 conditional-handle creation ────────────────────────────────────
//
// Mirrors `prism_gearbox_create_handle_ffi` (adjudicator.cu:1119).
// Default launch value = 0 (PRISM_ADJ_PRUNE) — see ticket §3 row 1.
//
// Called DURING parent-graph build (post-cuStreamEndCapture for the
// child template; before the SWITCH wire-in) so the bridge kernel-node
// can capture the handle as a kernel-node arg.
extern "C" int prism_f1_create_handle_ffi(
    cudaGraph_t   graph,
    uint32_t      default_value,
    uint64_t*     out_handle)
{
    if (graph == nullptr || out_handle == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaGraphConditionalHandle handle = 0;
    cudaError_t err = cudaGraphConditionalHandleCreate(
        &handle, graph,
        /*defaultLaunchValue=*/ default_value,
        /*flags=*/              cudaGraphCondAssignDefault);
    if (err != cudaSuccess) return static_cast<int>(err);
    *out_handle = static_cast<uint64_t>(handle);
    return static_cast<int>(cudaSuccess);
}

// ─── F1 SWITCH wire-in (size=3, parent-graph-resident) ─────────────────
//
// Mirrors `prism_gearbox_wire_with_handle_ffi` (adjudicator.cu:1137)
// with two key differences:
//   1. `nodeParams.conditional.size = 3u` (not 4) — F1 has 3 branches.
//   2. `out_body_subgraphs` is a 3-element array.
//
// **Hard-coded `cudaGraphCondTypeSwitch`** — DO NOT parameterize this
// to While. F1's case 2 (Violation) is a one-shot trap; making the
// SWITCH a While-loop would convert the trap into an unbounded retry
// (R5 in ticket §9). WHILE work is owned by separate Commits 8/9 in
// `while_drain_bridge.cu`.
//
// `predicate_node` becomes the SWITCH's only dependency, ensuring the
// SWITCH only fires AFTER the bridge kernel's `cudaGraphSetConditional`
// has executed.
extern "C" int prism_f1_wire_with_handle_ffi(
    cudaGraph_t      graph,
    cudaGraphNode_t  predicate_node,
    uint64_t         handle_v,
    cudaGraphNode_t* out_conditional_node,
    cudaGraph_t*     out_body_subgraphs    /* [3] */)
{
    if (graph == nullptr ||
        out_conditional_node == nullptr ||
        out_body_subgraphs == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaGraphConditionalHandle handle =
        static_cast<cudaGraphConditionalHandle>(handle_v);

    for (int i = 0; i < 3; ++i) out_body_subgraphs[i] = nullptr;

    cudaGraphNodeParams nodeParams{};
    nodeParams.type                 = cudaGraphNodeTypeConditional;
    nodeParams.conditional.handle   = handle;
    nodeParams.conditional.type     = cudaGraphCondTypeSwitch;
    nodeParams.conditional.size     = 3u;  // R5 regression guard — never While, never 4
    nodeParams.conditional.ctx      = nullptr;

    // Downstream of predicate_node so the SWITCH only fires after the
    // bridge kernel has called cudaGraphSetConditional.
    cudaError_t err = cudaGraphAddNode(
        out_conditional_node, graph,
        /*pDependencies=*/   &predicate_node,
        /*dependencyData=*/  nullptr,
        /*numDependencies=*/ 1u,
        &nodeParams);
    if (err != cudaSuccess) return static_cast<int>(err);

    // Driver-probe-confirmed (per G26 line 1173): phGraph_out is
    // populated for SWITCH-type on CUDA 13.x. Copy handles for caller.
    for (int i = 0; i < 3; ++i) {
        out_body_subgraphs[i] = nodeParams.conditional.phGraph_out
                                ? nodeParams.conditional.phGraph_out[i]
                                : nullptr;
    }
    // Defensive: a null body handle here means the runtime did not
    // populate phGraph_out for size=3 SWITCH — the populator below
    // will then surface cudaErrorInvalidValue (the operator-mandated
    // "Smoking Gun" signal, per the gearbox §B.3-narrow rationale).

    return static_cast<int>(cudaSuccess);
}

// ─── F1 SWITCH body populator ──────────────────────────────────────────
//
// Populates the 3 body sub-graphs returned by
// `prism_f1_wire_with_handle_ffi`:
//
//   body[0] = Prune     : EMPTY — no kernel, no memcpy. CUDA accepts
//                         empty body sub-graphs for SWITCH; the SWITCH
//                         reduces to "no-op" when this case is taken.
//                         Production flow continues to G26 SWITCH
//                         downstream of the F1 SWITCH.
//
//   body[1] = Construct : EMPTY — same rationale; the construct path
//                         is the ordinary downstream flow. ASC parent-
//                         side work (Directive Commits 10/11) will
//                         later populate this body when ASC lands.
//
//   body[2] = Violation : PTX trap kernel — halts the stream so the
//                         host sees `cudaErrorLaunchFailure` on the
//                         next sync. Short-circuits BEFORE G26 selects
//                         a gear (per ticket §2 justification 3).
//
// Empty bodies are intentional. The populator validates the handles
// are non-null (Smoking-Gun signal for the runtime not populating
// phGraph_out) and adds the trap kernel to body[2] only.
//
// Mirrors `prism_gearbox_populate_switch_bodies_ffi` (gearbox.cu:654)
// shape but with size=3 + only the trap body populated.

namespace {

// Canonical kernel-node addition wrapper for the F1 populator. Mirrors
// the anonymous-namespace helper in gearbox.cu (line 630). Kept private
// to this TU to avoid ODR collision; nvcc gives this internal linkage
// inside the anonymous namespace.
__host__ cudaError_t prism_f1_add_kernel_node(
    cudaGraphNode_t*       out_node,
    cudaGraph_t            graph,
    const cudaGraphNode_t* deps,
    size_t                 num_deps,
    void*                  func,
    dim3                   grid,
    dim3                   block,
    void**                 args,
    size_t                 shared_bytes)
{
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

extern "C" int prism_f1_populate_switch_bodies_ffi(
    cudaGraph_t* body_subgraphs    /* [3] */)
{
    if (body_subgraphs == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // ── Smoking-Gun: assert all three sub-graph handles are valid. ──
    // Same rationale as gearbox.cu:674 — a null handle here means
    // CUDA 13.x's cudaGraphAddNode(CONDITIONAL) did NOT populate
    // phGraph_out for size=3 SWITCH, and the caller must pivot to the
    // kernel-conditional fallback path.
    for (int i = 0; i < 3; ++i) {
        if (body_subgraphs[i] == nullptr) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }

    // ── body[0] = Prune : intentionally empty no-op. ──
    // No kernel-node added. SWITCH case 0 returns immediately to the
    // parent graph's downstream nodes (G26 SWITCH).
    (void)body_subgraphs[0];

    // ── body[1] = Construct : intentionally empty no-op. ──
    // No kernel-node added. SWITCH case 1 returns immediately to the
    // parent graph's downstream nodes (G26 SWITCH). ASC parent-side
    // work (Directive Commits 10/11) is reserved to populate this body.
    (void)body_subgraphs[1];

    // ── body[2] = Violation : PTX trap kernel. ──
    {
        cudaGraph_t body = body_subgraphs[2];
        cudaGraphNode_t trap_node = nullptr;
        cudaError_t rc = prism_f1_add_kernel_node(
            &trap_node, body,
            /*deps=*/ nullptr, /*num_deps=*/ 0,
            reinterpret_cast<void*>(prism_f1_violation_trap_kernel),
            dim3(1, 1, 1), dim3(1, 1, 1),
            /*args=*/ nullptr, /*shared=*/ 0);
        if (rc != cudaSuccess) return static_cast<int>(rc);
    }

    return static_cast<int>(cudaSuccess);
}

// ─── F1 predicate-bridge host launch shim ──────────────────────────────
//
// Mirrors `prism_gearbox_launch_predicate_bridge` (gearbox.cu:465).
// Single-thread launch on the supplied stream — used both during
// capture (to emit the kernel-node into the parent graph as a captured
// node) and during direct host launches (test paths).
//
// Argument order matches the kernel signature:
//   (d_adjudication_code, handle_v, mask)
//
// `mask` is exposed to the caller so future variants (e.g., 1-bit
// fast-path) can adjust without recompiling the kernel; canonical
// production value is 0x3.
// ─── F1 predicate-bridge kernel (TRACED variant) ──────────────────────
//
// Mirrors `prism_f1_predicate_bridge_kernel` but additionally writes
// the predicate value, branch count, invocation count, and first-launch
// sentinel into a pinned-mapped `PrismBranchTrace` struct visible to
// the host. Pass `d_trace == nullptr` to disable trace writes — the
// kernel is then byte-equivalent to the legacy bridge.
//
// CHUNK13_CAPTURED_GRAPH_LAUNCH_HANG diagnostic per
// .prism_orchestration/POST_CHUNK_LOOP_TEARDOWN_STALL_REPORT.md and the
// follow-up operator directive (2026-05-06).
extern "C"
__global__ void prism_f1_predicate_bridge_kernel_traced(
    const uint32_t*             __restrict__   d_adjudication_code,
    cudaGraphConditionalHandle                  handle,
    uint32_t                                    mask,
    PrismBranchTrace*           __restrict__   d_trace)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t code;
    if (d_adjudication_code == nullptr) {
        cudaGraphSetConditional(handle, 0u);
        code = 0u;
    } else {
        code = (*d_adjudication_code) & mask;
        cudaGraphSetConditional(handle, code);
    }
    if (d_trace != nullptr) {
        atomicExch(&d_trace->f1_predicate_last, code);
        atomicAdd(&d_trace->f1_branch_count[code & 0x3u], 1u);
        atomicAdd(&d_trace->f1_bridge_invocations, 1u);
        atomicCAS(&d_trace->first_launch_seen, 0u, 1u);
    }
}

extern "C" int prism_f1_launch_predicate_bridge_traced(
    const uint32_t* d_adjudication_code,
    uint64_t        handle_v,
    uint32_t        mask,
    void*           stream,
    uint64_t        branch_trace_dev)
{
    cudaGraphConditionalHandle handle =
        static_cast<cudaGraphConditionalHandle>(handle_v);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    PrismBranchTrace* d_trace =
        reinterpret_cast<PrismBranchTrace*>(branch_trace_dev);
    prism_f1_predicate_bridge_kernel_traced<<<1, 1, 0, s>>>(
        d_adjudication_code, handle, mask, d_trace);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int prism_f1_launch_predicate_bridge(
    const uint32_t* d_adjudication_code,
    uint64_t        handle_v,
    uint32_t        mask,
    void*           stream)
{
    cudaGraphConditionalHandle handle =
        static_cast<cudaGraphConditionalHandle>(handle_v);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    prism_f1_predicate_bridge_kernel<<<1, 1, 0, s>>>(
        d_adjudication_code, handle, mask);
    return static_cast<int>(cudaGetLastError());
}

#else  // CUDART_VERSION < 12060

// Pre-CUDA-12.6 fallback stubs. Mirror the pattern at adjudicator.cu:967
// (legacy F1) and adjudicator.cu:1186 (G26). Production toolchain is
// always CUDA 13.x so these are dead code on real hardware.

extern "C" int prism_f1_create_handle_ffi(
    cudaGraph_t /*graph*/, uint32_t /*default_value*/, uint64_t* /*out_handle*/
) {
    return static_cast<int>(cudaErrorNotSupported);
}

extern "C" int prism_f1_wire_with_handle_ffi(
    cudaGraph_t      /*graph*/,
    cudaGraphNode_t  /*predicate_node*/,
    uint64_t         /*handle_v*/,
    cudaGraphNode_t* /*out_conditional_node*/,
    cudaGraph_t*     /*out_body_subgraphs*/
) {
    return static_cast<int>(cudaErrorNotSupported);
}

extern "C" int prism_f1_populate_switch_bodies_ffi(
    cudaGraph_t* /*body_subgraphs*/
) {
    return static_cast<int>(cudaErrorNotSupported);
}

extern "C" int prism_f1_launch_predicate_bridge(
    const uint32_t* /*d_adjudication_code*/,
    uint64_t        /*handle_v*/,
    uint32_t        /*mask*/,
    void*           /*stream*/
) {
    return static_cast<int>(cudaErrorNotSupported);
}

extern "C" int prism_f1_launch_predicate_bridge_traced(
    const uint32_t* /*d_adjudication_code*/,
    uint64_t        /*handle_v*/,
    uint32_t        /*mask*/,
    void*           /*stream*/,
    uint64_t        /*branch_trace_dev*/
) {
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12060

// ════════════════════════════════════════════════════════════════════
// T7 — Noise-floor calibration writeback
// ════════════════════════════════════════════════════════════════════
//
// Two stream-ordered cudaMemcpyAsync(H2D) into the adjudicator
// struct's noise-floor fields. No kernel launch — the runtime API
// itself is captured-graph-compatible (memcpy-async nodes).
//
// Field layout (per CSR-C invariants in interferometric_adjudicator.rs):
//   noise_floor_mu     [f32; 6]  offset  0..24    24 B
//   noise_floor_sigma  [f32; 6]  offset 24..48    24 B

extern "C" int prism_adj_set_noise_floor_constants(
    InterferometricAdjudicatorFfi* adj,
    const float*                   mu_host,
    const float*                   sigma_host,
    void*                          stream_v
) {
    if (adj == nullptr || mu_host == nullptr || sigma_host == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaStream_t stream = static_cast<cudaStream_t>(stream_v);

    // Step 1: μ[6] → adj->noise_floor_mu (struct offset 0).
    cudaError_t err = cudaMemcpyAsync(
        &adj->noise_floor_mu[0],
        mu_host,
        sizeof(float) * 6,
        cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    // Step 2: σ[6] → adj->noise_floor_sigma (struct offset 24).
    // Same stream ⇒ ordered after the μ copy.
    err = cudaMemcpyAsync(
        &adj->noise_floor_sigma[0],
        sigma_host,
        sizeof(float) * 6,
        cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    return static_cast<int>(cudaSuccess);
}

// R03: downstream adjudication publishes immutable controls for the following
// physical step. It never retroactively changes a completed force/kick pair.
#include "prism_asc_decision.cuh"
extern "C" int prism_gearbox_stream_drain_v2(void*,uint32_t);
__global__ void prism_asc_init_v1(PrismAscDeviceV1* d,PrismAscDeviceV1 initial) {
    if(threadIdx.x||blockIdx.x)return;*d=initial;
    d->rows=reinterpret_cast<PrismAscAtomV1*>(d+1);
    prism_asc_initialize_work_tail(d);
}
__global__ void prism_asc_begin_v1(PrismAscDeviceV1* d) {
    if(threadIdx.x||blockIdx.x)return;
    if(!prism_asc_promote(*d)){d->status=ASC_BAD_DECISION;asm volatile("trap;");return;}
    d->phases[0]={};d->phases[1]={};d->control_work=0;d->coverage_failures=0;
}
__global__ void prism_asc_compose_v1(PrismAscDeviceV1* d,uint32_t phase,const float* x,float* f,double* pe,double* work) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=d->n_atoms)return;
    prism_asc_compose(d,i,d->n_atoms,phase,x+3*i,f+3*i,pe?pe+i:nullptr,work?work+i:nullptr);
}
__device__ __forceinline__ bool prism_asc_row_covered_v1(
    const PrismAscAtomV1& r, uint64_t physical_epoch, uint64_t decision_epoch) {
    bool covered = true;
    for (int k=0;k<4;++k)
        covered &= r.epoch[k]==physical_epoch && r.decision_epoch[k]==decision_epoch;
    return covered;
}
__device__ __forceinline__ double* prism_asc_phase_scalar_v1(PrismAscPhaseV1& s, uint32_t column) {
    if(column<3)return &s.force[column];
    if(column<6)return &s.torque[column-3];
    if(column<9)return &s.impulse[column-6];
    if(column<12)return &s.velocity_delta[column-9];
    return column==12?&s.differential_kick_energy:&s.potential;
}
__device__ __forceinline__ double prism_asc_row_scalar_v1(
    const PrismAscAtomV1& r, uint32_t phase, uint32_t column) {
    if(column<3)return r.force[phase][column];
    if(column<6)return r.torque[phase][column-3];
    if(column<9)return r.impulse[phase][column-6];
    if(column<12)return r.velocity_delta[phase][column-9];
    return column==12?r.differential_kick_energy[phase]:r.potential[phase];
}
__global__ void prism_asc_seal_v1(PrismAscDeviceV1* d) {
    // One block validates every receipt in parallel. The zero path is proved
    // from actual row values, never inferred from controller mode/strength:
    // stale coverage, NaNs and unexpected steering still reach the same checks.
    __shared__ uint32_t covered_rows[256], nonzero_rows[256];
    const uint32_t lane = threadIdx.x;
    const uint64_t epoch = d->physical_epoch, decision = d->active.producer_epoch;
    uint32_t covered_count = 0, nonzero = 0;
    for(uint64_t i=lane;i<d->n_atoms;i+=256){
        const auto& r=d->rows[i];
        if(!prism_asc_row_covered_v1(r,epoch,decision))continue;
        ++covered_count;
        bool nz=r.control_work!=0.0;
        for(int p=0;p<2;++p){
            nz |= r.potential[p]!=0.0 || r.differential_kick_energy[p]!=0.0;
            for(int j=0;j<3;++j)
                nz |= r.force[p][j]!=0.0 || r.torque[p][j]!=0.0 ||
                      r.impulse[p][j]!=0.0 || r.velocity_delta[p][j]!=0.0;
        }
        nonzero |= static_cast<uint32_t>(nz);
    }
    covered_rows[lane]=covered_count;nonzero_rows[lane]=nonzero;__syncthreads();
    // Integer-only tree reduction: no floating sum is reassociated.
    for(uint32_t stride=128;stride;stride>>=1){
        if(lane<stride){covered_rows[lane]+=covered_rows[lane+stride];nonzero_rows[lane]|=nonzero_rows[lane+stride];}
        __syncthreads();
    }
    if(nonzero_rows[0]==0){
        if(lane==0 && covered_rows[0]!=0){
            // Adding zero once equals the original repeated additions, also
            // preserving normalization of a pre-existing negative zero.
            d->control_work+=0.0;
            for(int p=0;p<2;++p)
                for(uint32_t c=0;c<14;++c)*prism_asc_phase_scalar_v1(d->phases[p],c)+=0.0;
        }
    }else{
        // Parallelize independent scalar columns, NOT atoms within a sum.
        // Every scalar retains the original increasing-atom addition order
        // and its existing accumulator value, including on error closeout.
        if(lane==0){
            double sum=d->control_work;
            for(uint32_t i=0;i<d->n_atoms;++i){const auto&r=d->rows[i];if(prism_asc_row_covered_v1(r,epoch,decision))sum+=r.control_work;}
            d->control_work=sum;
        }else if(lane<=28){
            const uint32_t p=(lane-1)/14,c=(lane-1)%14;
            double* out=prism_asc_phase_scalar_v1(d->phases[p],c);
            double sum=*out;
            for(uint32_t i=0;i<d->n_atoms;++i){const auto&r=d->rows[i];if(prism_asc_row_covered_v1(r,epoch,decision))sum+=prism_asc_row_scalar_v1(r,p,c);}
            *out=sum;
        }else if(lane<=30){
            const uint32_t p=lane-29;
            uint64_t nf_count=0,ni_count=0;
            for(uint32_t i=0;i<d->n_atoms;++i){
                const auto&r=d->rows[i];if(!prism_asc_row_covered_v1(r,epoch,decision))continue;
                bool nf=false,ni=false;
                for(int j=0;j<3;++j){nf|=r.force[p][j]!=0;ni|=r.impulse[p][j]!=0;}
                nf_count+=nf;ni_count+=ni;
            }
            d->phases[p].nonzero_force_atoms+=nf_count;
            d->phases[p].nonzero_impulse_atoms+=ni_count;
        }
    }
    __syncthreads();
    if(lane!=0)return;
    d->coverage_failures+=static_cast<uint64_t>(d->n_atoms)-covered_rows[0];
    for(int p=0;p<2;++p){d->phases[p].composed_atoms+=covered_rows[0];d->phases[p].consumed_atoms+=covered_rows[0];}
    bool finite=isfinite(d->control_work);
    for(int p=0;p<2;++p){const auto&s=d->phases[p];finite &= isfinite(s.potential)&&isfinite(s.differential_kick_energy);for(int j=0;j<3;++j)finite &= isfinite(s.force[j])&&isfinite(s.torque[j])&&isfinite(s.impulse[j])&&isfinite(s.velocity_delta[j]);}
    if(d->coverage_failures || !finite){d->status=d->coverage_failures?ASC_INCOMPLETE_COVERAGE:ASC_NONFINITE;asm volatile("trap;");return;}
    prism_asc_commit_control_work(d);
    d->completed_epoch=d->physical_epoch;++d->composed_steps;++d->consumed_steps;
    d->status=d->active.producer_epoch?ASC_COMPLETE:ASC_NO_PRIOR_DECISION;
}
__global__ void prism_asc_publish_v1(PrismAscDeviceV1* d,const InterferometricAdjudicatorFfi* adj) {
    if(threadIdx.x||blockIdx.x)return;
    if(d->physical_epoch==0 || d->completed_epoch!=d->physical_epoch || d->published_epoch>=d->physical_epoch){d->status=ASC_BAD_DECISION;asm volatile("trap;");return;}
    PrismAscControlV1 c={};c.source=d->source;c.mode=d->mode;c.generation=d->generation;
    c.producer_epoch=d->completed_epoch;c.producer_frame=d->completed_epoch;c.support_identity=1;c.status=ASC_COMPLETE;
    for(int j=0;j<4;++j)c.topology[j]=d->topology[j];
    if(adj->adjudication_code>PRISM_ADJ_VIOLATION || !isfinite(adj->current_divergence)) {d->status=ASC_BAD_DECISION;asm volatile("trap;");return;}
    if(adj->adjudication_code==PRISM_ADJ_CONSTRUCT){
        const auto* tile=adj->relaxed_manifold_ptr;
        if(!tile){d->status=ASC_BAD_DECISION;asm volatile("trap;");return;}
        c.valid=1;c.strength=d->alpha*adj->current_divergence;
        for(int j=0;j<3;++j)c.center[j]=0.5f*(tile->aabb_min[j]+tile->aabb_max[j]);
    }
    if(!prism_asc_control_matches(c,*d,d->completed_epoch)){d->status=ASC_BAD_DECISION;asm volatile("trap;");return;}
    d->pending=c;d->published_epoch=c.producer_epoch;++d->decisions_produced;
}
extern "C" int prism_asc_create_v1(uint64_t* out,uint32_t source,uint64_t generation,uint32_t atoms,
                                   const uint64_t* topology,float alpha,uint32_t mode,void* stream) {
    if(!out||!topology||!stream||!atoms||!generation||!isfinite(alpha)||mode!=ASC_LEGACY_WHOLE_SYSTEM)return (int)cudaErrorInvalidValue;
    *out=0;cudaStreamCaptureStatus cap;auto st=(cudaStream_t)stream;auto rc=cudaStreamIsCapturing(st,&cap);
    if(rc!=cudaSuccess)return (int)rc;if(cap!=cudaStreamCaptureStatusNone)return (int)cudaErrorStreamCaptureUnsupported;
    if(atoms>(SIZE_MAX-sizeof(PrismAscDeviceV1)-sizeof(PrismAscWorkCumulativeV1))/sizeof(PrismAscAtomV1))return (int)cudaErrorInvalidValue;
    PrismAscDeviceV1* d=nullptr;size_t bytes=sizeof(*d)+(size_t)atoms*sizeof(PrismAscAtomV1)+sizeof(PrismAscWorkCumulativeV1);
    rc=cudaMalloc((void**)&d,bytes);if(rc!=cudaSuccess)return (int)rc;*out=(uint64_t)d;
    rc=cudaMemsetAsync(d,0,bytes,st);if(rc!=cudaSuccess)return (int)rc;
    PrismAscDeviceV1 init={};init.abi=1;init.byte_size=sizeof(init);init.source=source;init.n_atoms=atoms;
    init.generation=generation;init.alpha=alpha;init.mode=mode;for(int j=0;j<4;++j)init.topology[j]=topology[j];
    prism_asc_init_v1<<<1,1,0,st>>>(d,init);return (int)cudaGetLastError();
}
extern "C" int prism_asc_begin_step_v1(uint64_t ptr,void* stream){if(!ptr||!stream)return 1;prism_asc_begin_v1<<<1,1,0,(cudaStream_t)stream>>>((PrismAscDeviceV1*)ptr);return (int)cudaGetLastError();}
extern "C" int prism_asc_seal_step_v1(uint64_t ptr,void* stream){if(!ptr||!stream)return 1;prism_asc_seal_v1<<<1,256,0,(cudaStream_t)stream>>>((PrismAscDeviceV1*)ptr);return (int)cudaGetLastError();}
extern "C" int prism_asc_publish_decision_v1(uint64_t ptr,const InterferometricAdjudicatorFfi* adj,void* stream){if(!ptr||!adj||!stream)return 1;prism_asc_publish_v1<<<1,1,0,(cudaStream_t)stream>>>((PrismAscDeviceV1*)ptr,adj);return (int)cudaGetLastError();}
extern "C" int prism_asc_compose_force_v1(uint64_t ptr,uint32_t n,uint32_t phase,const float* x,float* f,double* pe,double* work,void* stream){
    if(!ptr||!n||phase>1||!x||!f||!pe||!work||!stream)return 1;
    prism_asc_compose_v1<<<(n+255)/256,256,0,(cudaStream_t)stream>>>((PrismAscDeviceV1*)ptr,phase,x,f,pe,work);return (int)cudaGetLastError();
}
extern "C" int prism_asc_snapshot_v1(uint64_t ptr,void* stream,PrismAscDeviceV1* out){
    if(!ptr||!out||!stream)return 1;int rc=prism_gearbox_stream_drain_v2(stream,30000);if(rc)return rc;
    return (int)cudaMemcpy(out,(void*)ptr,sizeof(*out),cudaMemcpyDeviceToHost);
}
// New symbol makes mixed old/new CUDA objects fail at link time. Measurement
// copies execute after one bounded drain; they never mutate physical state.
extern "C" int prism_asc_measurement_snapshot_v1(uint64_t ptr,void* stream,
    PrismAscDeviceV1* out,PrismAscWorkCumulativeV1* work,void* rows,size_t row_bytes) {
    if(!ptr||!stream||!out||!work||!rows)return (int)cudaErrorInvalidValue;
    int dr=prism_gearbox_stream_drain_v2(stream,30000);if(dr)return dr;
    auto rc=cudaMemcpy(out,(void*)ptr,sizeof(*out),cudaMemcpyDeviceToHost);
    if(rc!=cudaSuccess)return (int)rc;
    if(out->abi!=1||out->byte_size!=sizeof(*out)||!out->rows ||
       out->n_atoms>SIZE_MAX/sizeof(PrismAscAtomV1)||
       row_bytes!=(size_t)out->n_atoms*sizeof(PrismAscAtomV1))return (int)cudaErrorInvalidValue;
    rc=cudaMemcpy(work,out->rows+out->n_atoms,sizeof(*work),cudaMemcpyDeviceToHost);
    if(rc!=cudaSuccess)return (int)rc;
    if(work->magic!=PRISM_ASC_WORK_MAGIC_V1)return (int)cudaErrorInvalidValue;
    return (int)cudaMemcpy(rows,out->rows,row_bytes,cudaMemcpyDeviceToHost);
}
extern "C" int prism_asc_destroy_v1(uint64_t ptr,void* stream){
    if(!ptr)return 0;int rc=prism_gearbox_stream_drain_v2(stream,30000);if(rc)return rc;return (int)cudaFree((void*)ptr);
}

extern "C" int prism_graph_rebind_handle_arguments_v1(cudaGraph_t,uint64_t,uint64_t,
    void* const*,const uint32_t*,uint32_t,uint32_t,uint32_t*);
extern "C" int prism_f1_rebind_parent_predicates_v1(cudaGraph_t graph,uint64_t old_handle,uint64_t new_handle,uint32_t* changed) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040
    void* functions[]={reinterpret_cast<void*>(prism_f1_predicate_bridge_kernel),reinterpret_cast<void*>(prism_f1_predicate_bridge_kernel_traced)};
    const uint32_t counts[]={3,4};
    return prism_graph_rebind_handle_arguments_v1(graph,old_handle,new_handle,functions,counts,2,1,changed);
#else
    return static_cast<int>(cudaErrorNotSupported);
#endif
}

// Exact module state transfer at an all-worker quiescent boundary.
// Caller owns the context, holds every producer and synchronizes before entry.
extern "C" size_t prism_checkpoint_adjudicator_bytes() { return sizeof(c_mu_01_sq); }
extern "C" int prism_checkpoint_adjudicator_capture(unsigned char* bytes, size_t length, cudaStream_t stream) {
    if (!bytes || length != prism_checkpoint_adjudicator_bytes()) return (int)cudaErrorInvalidValue;
    size_t offset = 0; cudaError_t rc;
    rc = cudaMemcpyFromSymbolAsync(bytes + offset, c_mu_01_sq, sizeof(c_mu_01_sq), 0, cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(c_mu_01_sq);
    return (int)cudaStreamSynchronize(stream);
}
extern "C" int prism_checkpoint_adjudicator_restore(const unsigned char* bytes, size_t length, cudaStream_t stream) {
    if (!bytes || length != prism_checkpoint_adjudicator_bytes()) return (int)cudaErrorInvalidValue;
    size_t offset = 0; cudaError_t rc;
    rc = cudaMemcpyToSymbolAsync(c_mu_01_sq, bytes + offset, sizeof(c_mu_01_sq), 0, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(c_mu_01_sq);
    return (int)cudaStreamSynchronize(stream);
}
