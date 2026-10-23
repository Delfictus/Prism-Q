//! # Interferometric Adjudicator — the F1 SWITCH brain
//!
//! Implements the **Brain** of the PRISM-4D Digital Spatiotemporal
//! Interferometer (per the Claude-2 Architectural Mandate, Blackwell
//! Convergence Project). The Adjudicator computes the Kullback-Leibler
//! divergence Δ_AB between the Relaxed (P) and UV-Perturbed (Q) SO(3)
//! power spectra produced by Claude-1's WMMA tensor-core kernels
//! ([`crate::so3_project`]), thresholds the result against a running
//! 3-sigma thermal noise floor, and writes a 32-bit `adjudication_code`
//! into pinned device memory. The Blackwell `sm_120` hardware
//! scheduler reads that code and routes the F1 `cudaGraphConditionalNode`
//! to one of three sub-graphs:
//!
//! | code | route                             | trigger                                       |
//! |------|-----------------------------------|-----------------------------------------------|
//! | `0`  | Prune Noise                       | Δ_AB ≤ μ_noise + 3σ                            |
//! | `1`  | Burst Detected — heavy bisimulation| Δ_AB > μ_noise + 3σ                            |
//! | `2`  | Hard Trap — invariant violation   | NaN / Inf / non-positive ratio in log argument|
//!
//! The Adjudicator is a **Total Function**: every 32-bit bit-pattern
//! at the kernel input maps to exactly one valid `adjudication_code`
//! ∈ {0, 1, 2}. Inline PTX guards (see [`Self::APPLY_GUARD_DOC`])
//! prevent any `NaN` from leaving the kernel.
//!
//! ## Memory layout — 256-byte zero-trust lock (Emergency Rectification 2026-05-02)
//!
//! Per CSR-section C of the mandate, `sizeof::<InterferometricAdjudicatorFfi>()`
//! is **256 bytes exactly** (two Blackwell L1 sectors).  The operator's
//! Emergency Kill / Zero-Trust Alignment directive reverts `gear_override`
//! to its B.3.2 home at offset 100 and renames two fields for ABI clarity
//! (`force_prune_mask` → `force_prune_mask_ptr`; `d_potential_energy` →
//! `potential_energy`).  `d_external_work` retains its M1.2.18.5 pointer-
//! fusion semantics (`*mut f64` at offset 128) so multi-kernel asynchronous
//! atomicAdd-f64 contributions (UV kicks, ASC, force/velocity clamps) land
//! in a single F2-pool address — no read-modify-write hazard.
//!
//! ```text
//! noise_floor_mu          [f32; 6]  offset   0..24    24 B
//! noise_floor_sigma       [f32; 6]  offset  24..48    24 B
//! current_divergence      f32       offset  48..52     4 B
//! adjudication_code       u32       offset  52..56     4 B  (pre_rank::AdjudicationCode-compatible)
//! relaxed_manifold_ptr    ptr       offset  56..64     8 B
//! perturbed_manifold_ptr  ptr       offset  64..72     8 B
//! start_clock             u64       offset  72..80     8 B
//! stop_clock              u64       offset  80..88     8 B
//! legacy_centroid_fallback [f32;3]  offset  88..100   12 B  (Anti-Greenfield § 6.2)
//! gear_override           u32       offset 100..104    4 B  (B.3.2 Hardware Interlock — RESTORED)
//! force_prune_mask_ptr    *mut u64  offset 104..112    8 B  (G28 SISR)
//! potential_energy        f64       offset 112..120    8 B  (M1.2.17 — Hamiltonian scalar VALUE)
//! d_dt                    *mut f32  offset 120..128    8 B  (Gearbox dt target)
//! d_external_work         *mut f64  offset 128..136    8 B  (W_ext POINTER, F2-pool)
//! _trailing_pad           ...      offset 136..256  120 B  (implicit via #[repr(C, align(128))])
//! ─────────────────────────────────────────────────────────
//! TOTAL                                                256 B
//! ```
//!
//! ## M1.2.17 — Layout pivot (operator unified directive 2026-05-02)
//!
//! `d_velocities` removed from the struct (was at offset 120 in B.3.2).
//! It remains a `PipelineConfig` field — the gearbox SWITCH bodies'
//! velocity-rescale kernel reads it through populator FFI plumbing,
//! not through the FFI struct.
//!
//! `d_potential_energy` is the system-wide potential energy SCALAR
//! VALUE (not a pointer) at offset 112.  The CUB DeviceReduce::Sum
//! captured node writes it every step from the per-atom AMBER PE
//! components buffer.  The SFA kernel reads it for the 1% drift
//! Hamiltonian Stability Fuse → Gear 3 trap on instability.
//!
//! `d_dt` moves to offset 120; T12 Pre-Flight pre-capture
//! `cuMemcpyHtoD` wire-up updated.
//!
//! ## `gear_override` Hardware Interlock (operator Zero-Trust 2026-05-02)
//!
//! `gear_override` is the **Manual Override Invariant** — a u32 at
//! offset **100** that the host (operator or safety script) writes via
//! `cuMemcpyHtoDAsync` to force a gear shift independent of the
//! KL-driven adjudication state machine.  Sentinel `0xFF` = Auto
//! (gearbox is free to choose); values 0..3 = manual force to that
//! gear.  The predicate-bridge kernel reads this field on every
//! captured-graph launch and short-circuits the SFA's calculated gear
//! when override is set.
//!
//! Slot history: B.3.2 placed `gear_override` at offset 100 (claiming
//! the implicit padding for `force_prune_mask_ptr`'s 8-byte alignment).
//! M1.2.18.5 attempted to move it to offset 136 to free that slot for
//! `_pad_100`; the Emergency Rectification reverted it to offset 100
//! per the operator's Forced-Gear Supervisor Shim spec which writes to
//! `(adj_dev + 100)` directly.
//!
//! ## T12 Pre-Flight — connective tissue (Wave A)
//!
//! `d_dt` and `d_velocities` are wired in for Wave B's G26 chronometric
//! gearbox: the SWITCH body sub-graphs will swap the integrator's dt
//! coefficient and rescale velocities on gear-transition frames.
//! Wave A leaves these pointers wired but UNCONSUMED — no kernel
//! reads them, no integrator depends on them. Pure surface prep.
//!
//! ## Anti-Greenfield § 6.2 — `legacy_centroid_fallback`
//!
//! Stores a single `[f32; 3]` representative centroid that legacy
//! `nhs_rt_full.rs` site-logging can read without understanding the
//! full SO(3) manifold. Populated by the Adjudicator kernel from
//! `relaxed_manifold_ptr->aabb` mid-point at every step. This lets
//! the legacy diagnostic-script path observe site coordinates
//! through the same ABI it always used; new code reads the full
//! manifold via the typed pointers above. **Surgical extension, not
//! parallel type.**
//!
//! ## Anti-Greenfield § 5 — adjudication-code reuse
//!
//! The `adjudication_code` field stores values from the existing
//! [`crate::pre_rank::AdjudicationCode`] enum (RECT-2, commit
//! `0993bf9b`). T0 originally defined parallel `ADJUDICATION_PRUNE /
//! _BURST / _ABORT` constants on this type — those have been removed
//! per the Anti-Greenfield Scavenge-Before-Code rule. Decode raw u32
//! values via [`crate::pre_rank::AdjudicationCode::from_raw`].
//!
//! The compile-time `const _: () = { assert!(...) }` block at the foot
//! of this file enforces the invariant; the runtime test module
//! pins every field's byte-offset.
//!
//! ## F2 pool allocation requirement
//!
//! Per the F2 mandate, `InterferometricAdjudicatorFfi` instances MUST
//! be allocated via [`crate::vram_pool::VramPool::alloc_async`]
//! (`cudaMallocFromPoolAsync`) and remain pointer-stable for the
//! entire MD campaign. `cudaMalloc` and static `__device__` arrays
//! are **forbidden** — the former host-syncs (breaking captured-graph
//! replay), the latter limits multi-stream scalability and would
//! exceed the 164 KB shared-memory limit on `sm_120`.
//!
//! ## F1 race-condition prevention (CSR-D)
//!
//! Within the Adjudicator kernel, the writes to `current_divergence`
//! and `adjudication_code` happen on the same single thread in
//! program order. A `__threadfence()` immediately before the
//! `adjudication_code` store flushes the divergence write to L2
//! before the SWITCH-node read can observe a stale value. The kernel
//! launches as `<<<1, 1, 0, stream>>>` so the only ordering question
//! is between the kernel's exit and the SWITCH-node's next read,
//! which the captured-graph topology already serialises (the SWITCH
//! node is downstream of the Adjudicator node in the same stream).
//!
//! ## Wavelength encoding — open question for T1
//!
//! The mandate's T1 spec calls for ingesting `wavelength_nm` from
//! [`crate::rich_spike::RichSpike`]. The CLA-1 schema as committed
//! at `40c4db10` does **not** include a `wavelength_nm` field. Two
//! resolution paths are tractable without an FFI break:
//!
//!   1. Pack the 4 authorised wavelengths into the
//!      `RichSpike::spike_source` enum (currently `LIF / UV / EFP /
//!      LADD / COFIRE`) by adding `UV_260`, `UV_280`, `UV_305`,
//!      `UV_320` discriminants.
//!   2. Pack the wavelength index (2 bits) into the high bits of
//!      `RichSpike::intensity_packed`.
//!
//! T1 is blocked on operator decision. Option (1) is preferred —
//! the wavelength is a property of the perturbation source, and
//! the source enum is the natural carrier.

#![allow(dead_code)]

use crate::so3_project::ContactShellTile;

// ============================================================================
// InterferometricAdjudicatorFfi — 128-byte FFI struct (one Blackwell L1 sector)
// ============================================================================

/// Per-stream Adjudicator state. Allocated once via the F2 pool at
/// MD-campaign init; pointer-stable for the entire campaign.
///
/// **Layout invariants** — enforced at compile time below; see
/// module-level docs for the full byte-offset table.
///
/// * `sizeof == 128`
/// * `alignof == 128`
/// * Field order matches the C-side mirror in
///   `cuda/interferometric_adjudicator.cuh` byte-for-byte.
#[repr(C, align(128))]
#[derive(Debug)]
pub struct InterferometricAdjudicatorFfi {
    /// Running mean of the thermal noise floor, per SH band
    /// (l = 0..5). Updated once per "Cool" frame by the Adjudicator
    /// kernel itself. **Offset 0 — must remain offset 0 for the
    /// Adjudicator kernel's `LDG.E.128` aligned-load path.**
    pub noise_floor_mu: [f32; 6],

    /// Running stddev of the thermal noise floor, per SH band. The
    /// Burst threshold is `mu + 3*sigma` per band (see kernel doc
    /// for the per-band-vs-aggregate decision).
    pub noise_floor_sigma: [f32; 6],

    /// Latest computed Σ KL-divergence Δ_AB between the relaxed
    /// (P) and perturbed (Q) `ContactShellTile`s. Written by the
    /// Adjudicator kernel **before** [`Self::adjudication_code`]
    /// (program order + `__threadfence()` ⇒ no race vs SWITCH-node
    /// read).
    pub current_divergence: f32,

    /// F1 SWITCH selector. **Offset 52** — pinned. The Blackwell
    /// hardware scheduler reads this 4-byte word to route the
    /// `cudaGraphConditionalNode`. Values are
    /// [`crate::pre_rank::AdjudicationCode`]-compatible:
    ///
    /// * `0` — `Prune` (Δ_AB below 3-σ threshold; site dropped)
    /// * `1` — `Construct` (Δ_AB above threshold; bisimulation path)
    /// * `2` — `Violation` (NaN / Inf / non-positive ratio; SAD-PATH abort)
    ///
    /// **The Adjudicator is a Total Function**: any 32-bit input
    /// pattern produces exactly one of these three values. The
    /// inline PTX guards in the kernel ensure no NaN escapes.
    /// Decode via `AdjudicationCode::from_raw(self.adjudication_code)`.
    pub adjudication_code: u32,

    /// F2-pool-allocated pointer to the relaxed-state manifold (P).
    /// Lifetime: the `ContactShellTile` referenced here must outlive
    /// every Adjudicator-kernel launch in the captured graph.
    pub relaxed_manifold_ptr: *const ContactShellTile,

    /// F2-pool-allocated pointer to the perturbed-state manifold (Q).
    /// Same lifetime contract as `relaxed_manifold_ptr`.
    pub perturbed_manifold_ptr: *const ContactShellTile,

    /// Hardware clock at Adjudicator-kernel entry, captured via
    /// `clock64()` PTX. Used by the T4 telemetry harness to verify
    /// the < 10 μs ASC-update latency invariant (CSR-H section).
    pub start_clock: u64,

    /// Hardware clock at Adjudicator-kernel exit. `(stop - start)`
    /// in ns is exported to `SiteManifest::hardware_telemetry` (T4).
    pub stop_clock: u64,

    /// **Anti-Greenfield § 6.2 backward-compatibility shim.** Single
    /// representative centroid (`x`, `y`, `z` in Å) the legacy
    /// `nhs_rt_full.rs` site-logging path can read through its
    /// existing 3-float ABI. Populated by the Adjudicator kernel
    /// from `relaxed_manifold_ptr->aabb` mid-point at every step;
    /// **never** participates in the F1 SWITCH decision. This lets
    /// legacy diagnostics observe coordinates without understanding
    /// the full SO(3) manifold.
    pub legacy_centroid_fallback: [f32; 3],

    /// **B.3.2 — Manual Gear Override (Hardware Interlock).**
    /// **Offset 100** (Emergency Rectification 2026-05-02 — restored from
    /// the M1.2.18.5 attempt to relocate to 136).  u32 the host writes
    /// via `cuMemcpyHtoDAsync` to force a gear shift, bypassing the
    /// KL-driven SFA.  Sentinel `0xFF` (i.e., 0x000000FF) = Auto;
    /// values 0..3 = manual force (0 = 0.5fs / 1 = 2.0fs / 2 = 4.0fs /
    /// 3 = abort).  Read on every captured-graph launch by the
    /// predicate-bridge kernel:
    ///
    /// ```text
    ///     final_gear = (override == 0xFF) ? calculated : (override & 0x03)
    /// ```
    pub gear_override: u32,

    /// **G28 SISR per-cluster prune-bit mask buffer pointer.**
    /// Offset 104..112, 8 B (`*mut u64`).  Pointer to a single `u64`
    /// in F2-pool device memory.  Bit `i` is set by the SISR kernel
    /// when site `i` fails the bilateral symmetry check (no Chain-B
    /// partner manifold within ε_sym of the C2-reflected AABB centroid).
    /// The Adjudicator step kernel ANDs this mask with
    /// `(1u64 << cluster_id)` and forces `adjudication_code = 0` (Prune)
    /// on a non-zero result, **independent of** Δ_AB magnitude.
    /// Null disables the symmetry gate (legacy / non-dimer targets).
    ///
    /// Renamed from `force_prune_mask` to `force_prune_mask_ptr` per
    /// the operator's Zero-Trust §1.1 — the `_ptr` suffix makes the
    /// pointer-vs-value distinction explicit at the FFI boundary.
    pub force_prune_mask_ptr: *mut u64,

    /// **M1.2.17 — Hamiltonian scalar (potential energy VALUE).**
    /// Offset 112..120, 8 B.  System-wide total potential energy in
    /// kcal/mol, written every step by the captured CUB
    /// DeviceReduce::Sum node from the per-atom AMBER PE components
    /// buffer.  This is a VALUE (not a pointer) — host code reads it
    /// directly from the F2-pool-resident FFI struct.
    ///
    /// Stability fuse: the SFA kernel reads `potential_energy` and
    /// compares against `cruise.v_prev`; if `|V_t − (V_{t-1} + W_ext)|
    /// / |V_{t-1} + W_ext| > 0.01` or `isnan(V_t)`, target_gear = 3
    /// (abort/trap).  See `gearbox.cu::prism_gearbox_sfa_kernel`.
    ///
    /// Renamed from `d_potential_energy` to `potential_energy` per the
    /// operator's Zero-Trust §1.1 — the `d_` prefix is reserved for
    /// pointer fields (`d_dt`, `d_external_work`); a VALUE field has
    /// no prefix.
    pub potential_energy: f64,

    /// **G26 chronometric gearbox dt pointer.**  Offset 120..128, 8 B.
    /// Device-resident `*mut f32` carrying the active integrator
    /// timestep (in ps).  The G26 SWITCH body sub-graphs (Gears 0/1/2)
    /// write to `*d_dt` via `apply_fixed_dt_kernel` — the integrator
    /// reads its dt from this address every step (`d_protocol->dt`
    /// wired by `NhsAmberFusedEngine::d_protocol_dt_dev_ptr()`).
    ///
    /// Was at offset 112 in B.3.2; M1.2.17 moved it to 120 to make
    /// room for `d_potential_energy` at the L1-aligned 112 slot.
    pub d_dt: *mut f32,

    /// **M1.2.18.5 — Total External Work POINTER (Hamiltonian audit).**
    /// Offset 128..136, 8 B (`*mut f64`).  VRAM-native accumulator for
    /// non-conservative energy injections that the SFA stability fuse
    /// must subtract from ΔV before computing drift:
    ///
    ///   * UV velocity kicks (apply_vibrational_transfer):
    ///     ΔK = ½·m·(v_new² − v_old²) f64-precision, atomicAdd_f64.
    ///   * Force/velocity clamps (M1.2.18 follow-up): clamp losses.
    ///
    /// ASC steering work is folded into `d_potential_energy` directly
    /// (operator §3.2 — accumulate V_ASC into pe_components so V_t
    /// already reflects the steering potential).
    ///
    /// Pointer-fusion rationale (M1.2.18.5):
    ///   * F2-pool `*mut f64` allocated once, address-stable for the
    ///     entire MD campaign.  The captured graph emits a
    ///     `cuMemsetD8Async` head-of-loop node that zeros `*d_external_work`
    ///     each replay so the SFA reads a fresh chunk-window value.
    ///   * `nhs_amber_fused_step`'s 70th parameter binds the same address.
    ///     `apply_vibrational_transfer` (was `nullptr` pre-M1.2.18.5)
    ///     now writes via `atomicAdd(d_external_work, delta_k)` —
    ///     sm_120 native f64 atomic.
    ///
    /// First-Law drift formula (M1.2.18.5 §2):
    ///   Drift = |V_t − (V_{t-1} + W_ext)| / |V_{t-1} + W_ext|
    ///   > 0.01 ⇒ Gear 3 abort trap.
    ///
    /// `null` is the default (e.g., test fixtures) — captured pipeline
    /// gates the W_ext read on a non-null pointer.
    pub d_external_work: *mut f64,

    /// **M1.2.20.C — "Stiffness" Handle.**  Offset 136..140, 4 B (`f32`).
    /// Base η for the gradient gasp kernel; per Ruling 2 it locks to
    /// 1.0 in the captured pipeline init.  Effective gain is
    /// `η_base / max(spike.vib_energy, 1e-3)` per spike, multiplied by
    /// `Q_s(λ)` from the UV LUT, multiplied by 10.0 if and only if the
    /// engine's current step counter equals `force_burst_step` (offset
    /// 140).  Host can rewrite at any time via `cuMemcpyHtoDAsync` on
    /// the captured-graph stream — no kernel restart required.
    pub gasp_gain_eta: f32,

    /// **M1.2.20.C — "Trigger" Handle.**  Offset 140..144, 4 B (`u32`).
    /// Step number at which the gasp kernel applies a 10× amplification
    /// to η_eff.  Sentinel `u32::MAX` = no burst (gasp runs at baseline
    /// gain every replay).  Wired to the `--force-burst-at-step` CLI
    /// flag; the orchestrator writes this once at engine init and the
    /// gasp kernel reads it every replay.
    pub force_burst_step: u32,

    /// **M1.2.20.C-B — PTX Momentum Guard flag.**  Offset 144..148,
    /// 4 B (`u32`).  Set to 1 by `prism_momentum_guard_check_kernel`
    /// when the gasp-induced center-of-mass translation
    /// `|Σ m_i · Δr_i|` exceeds 1e-4 Å (operator §3 Zero-Trust
    /// invariant — a legitimate gasp is an *expansion*, not a
    /// translation; if the protein "walks" during the perturbation
    /// the SO(3) power spectrum becomes ungrounded noise).  The
    /// Adjudicator step kernel reads this flag and forces
    /// `adjudication_code = PRISM_ADJ_VIOLATION` (code = 2),
    /// overriding the KL-divergence trigger.  Reset to 0 each
    /// captured-graph replay via cuMemsetD8Async at offset 144.
    pub momentum_violation_flag: u32,

    /// **M1.2.26.RECOUPLE Part 2 — KL-Native Noise Floor (mean).**
    /// Offset 148..152, 4 B (`f32`).  Run-locked μ of `total_kl`
    /// computed during the Phase-2 segment of the Split T7 epoch
    /// (steps 51-100, KL(P_frozen || Q_current) per cluster). Read by
    /// the F1 SWITCH predicate as the dimensionally-synchronized
    /// baseline:  threshold = total_kl_mu + 12.0f * total_kl_sigma.
    /// Sentinel `0.0f` ⇒ Phase 2 has not converged; threshold falls
    /// back to the legacy `noise_floor_mu[0] + 12σ` path for
    /// compatibility with non-KL-calibrated targets.
    pub total_kl_mu: f32,

    /// **M1.2.26.RECOUPLE Part 2 — KL-Native Noise Floor (stddev).**
    /// Offset 152..156, 4 B (`f32`).  Run-locked σ of `total_kl`
    /// computed during Phase 2 (steps 51-100). Pairs with
    /// `total_kl_mu` to form the KL-native discovery threshold.
    /// Sentinel `0.0f` ⇒ legacy fallback (see total_kl_mu doc).
    pub total_kl_sigma: f32,

    /// **M1.2.20.C-G / T24 — Adjudication Reason Flags.**  Offset
    /// 156..160, 4 B (`u32`).  Bit-mapped self-auditing field set by
    /// the gasp kernel + Adjudicator step kernel + Momentum Guard +
    /// SISR; read by host at teardown via cuMemcpyDtoH and stamped
    /// into `v2_ignition_summary.json`.  Replaces the M1.2.20.C-C
    /// `lqi_flags` (same semantic; relocated from offset 148 in
    /// M1.2.26.RECOUPLE to make room for total_kl_mu/sigma at
    /// 148/152). Bit-by-offset readers in `so3_project.cu` use the
    /// new offset 156.
    ///
    /// Bit-mapping (operator §4.1):
    ///   • Bit 0 (`0x1`)        — `NaN_POTENTIAL`  : non-finite operand
    ///   • Bit 1 (`0x2`)        — `MOMENTUM_VIOLATION`: |Σm·Δr| > 1e-4 Å
    ///   • Bit 2 (`0x4`)        — `SYMMETRY_VETO`  : SISR prune mask non-zero
    ///   • Bit 3 (`0x8`)        — `GAIN_SATURATION`: η_eff hit hard cap
    ///   • Bits 4..30           — reserved
    ///   • Bit 31 (`0x80000000`) — `LQI_T7_VARIANCE_ZERO` (Lineage
    ///                              Protection — kept from M1.2.20.C-C)
    pub adjudication_reason_flags: u32,

    /// **M1.2.26.RECOUPLE — Reserved padding.**  Offset 160..256, 96 B.
    /// Pads `InterferometricAdjudicatorFfi` to an explicit 256-byte
    /// dual-sector total. Renamed from `_reserved_m1_2_20` to
    /// `_reserved_m1_2_26` after total_kl_mu/sigma carved 8 B out
    /// of the prior 104-B reservation.
    pub _reserved_m1_2_26: [u8; 96],
}

impl InterferometricAdjudicatorFfi {
    /// All-zero initial state. Equivalent to the C-side
    /// `interferometric_adjudicator_zero_kernel`'s effect.
    /// `cluster_id`-style sentinels are not used — every numeric
    /// field has a meaningful zero (no allocations, no decisions
    /// made yet).
    ///
    /// `relaxed_manifold_ptr` and `perturbed_manifold_ptr` are
    /// initialised to `null`; the wire-in code (T2 integration)
    /// **must** populate them before the first kernel launch.
    pub const fn zero() -> Self {
        Self {
            noise_floor_mu: [0.0; 6],
            noise_floor_sigma: [0.0; 6],
            current_divergence: 0.0,
            adjudication_code: 0,
            relaxed_manifold_ptr: std::ptr::null(),
            perturbed_manifold_ptr: std::ptr::null(),
            start_clock: 0,
            stop_clock: 0,
            legacy_centroid_fallback: [0.0; 3],
            gear_override: Self::GEAR_OVERRIDE_AUTO,
            force_prune_mask_ptr: std::ptr::null_mut(),
            potential_energy: 0.0,
            d_dt: std::ptr::null_mut(),
            d_external_work: std::ptr::null_mut(),
            // M1.2.20.C-A — gasp handles default to neutral state.
            // η_base = 1.0 per Ruling 2; force_burst_step = u32::MAX
            // sentinel disables the 10× amplification.
            gasp_gain_eta: 1.0,
            force_burst_step: Self::FORCE_BURST_DISABLED,
            momentum_violation_flag: 0,
            // M1.2.26.RECOUPLE Part 2 — KL-native μ/σ start zero.
            // Phase 2 of the Split T7 Epoch (steps 51-100) writes
            // these. Until then the F1 SWITCH falls back to the
            // legacy noise_floor_mu[0] threshold.
            total_kl_mu: 0.0,
            total_kl_sigma: 0.0,
            adjudication_reason_flags: 0,
            _reserved_m1_2_26: [0u8; 96],
        }
    }

    /// **M1.2.20.C-G / T24** — bit-map for `adjudication_reason_flags`.
    pub const REASON_NAN_POTENTIAL: u32 = 0x0000_0001;
    pub const REASON_MOMENTUM_VIOLATION: u32 = 0x0000_0002;
    pub const REASON_SYMMETRY_VETO: u32 = 0x0000_0004;
    pub const REASON_GAIN_SATURATION: u32 = 0x0000_0008;
    /// Lineage Protection bit retained from M1.2.20.C-C.  Bit-31 of
    /// `adjudication_reason_flags` indicates the T7 reduce kernel
    /// found σ² <= 0 after 100 cold-hold samples.
    pub const LQI_T7_VARIANCE_ZERO: u32 = 0x8000_0000;

    /// **M1.2.20.C** — sentinel value for `force_burst_step` meaning
    /// "no burst scheduled" (gasp runs at baseline gain every replay).
    /// Equal to `u32::MAX = 0xFFFFFFFFu`.
    pub const FORCE_BURST_DISABLED: u32 = u32::MAX;

    /// B.3.2 — sentinel value for `gear_override` meaning "Auto"
    /// (the gearbox's KL-driven SFA decides the gear).  Stored as
    /// u32 = 0x000000FF — the predicate bridge compares the full
    /// word against this constant.  Anything other than the sentinel
    /// is a manual force; only the low 2 bits are honoured (& 0x03).
    pub const GEAR_OVERRIDE_AUTO: u32 = 0xFFu32;

    /// Documented PTX guard sequence used by the Adjudicator kernel
    /// before every `flog2.approx`. Reproduced here as a string
    /// constant so the Rust-side test of CSR-section B can verify
    /// the .cu file matches.
    ///
    /// Failure modes routed to `ADJUDICATION_ABORT`:
    ///   - `setp.nan.f32 p, %ratio` — NaN input.
    ///   - `setp.le.f32 p, %ratio, 0f00000000` — non-positive ratio.
    ///   - `setp.gtu.f32 p, %ratio, 0f7f800000` — ratio > +Inf
    ///     (sentinel; never fires under IEEE 754).
    pub const APPLY_GUARD_DOC: &'static str = "\
        setp.nan.f32 p_nan, %ratio;\n\
        setp.le.f32  p_le0, %ratio, 0f00000000;\n\
        @p_nan mov.u32 %code, 2;\n\
        @p_le0 mov.u32 %code, 2;\n\
        @p_nan @p_le0 bra ABORT_PATH;\n";
}

// SAFETY: `InterferometricAdjudicatorFfi` is a POD struct whose
// pointer fields are owned by the F2 pool and managed by the
// kernel-launch contract — never dereferenced from Rust on the host
// side. Send + Sync are sound because every field is either Copy or
// a raw pointer the host treats as opaque.
unsafe impl Send for InterferometricAdjudicatorFfi {}
unsafe impl Sync for InterferometricAdjudicatorFfi {}

// ============================================================================
// Compile-time layout invariants (CSR-C)
// ============================================================================

const _: () = {
    use std::mem::{align_of, offset_of, size_of};

    // Emergency Rectification 2026-05-02 — Zero-Trust 256-byte lock.
    // gear_override @ 100 (B.3.2 home), force_prune_mask_ptr @ 104,
    // potential_energy @ 112, d_dt @ 120, d_external_work @ 128.
    // Trailing 120 B implicit pad (#[repr(C, align(128))] rounds up
    // 136 used → 256 physical).  Byte-offset asserts below pin every
    // FFI-visible field; drift here breaks the C++ static_asserts in
    // adjudicator.cuh and the byte-offset reads in gearbox.cu.
    assert!(size_of::<InterferometricAdjudicatorFfi>() == 256);

    // alignof == 128 — Blackwell L1 sector boundary preserved.
    assert!(align_of::<InterferometricAdjudicatorFfi>() == 128);

    // The pointer width must be 8 — the byte-offset table assumes
    // a 64-bit target.  Any target where size_of::<*const T>() != 8
    // would need a layout audit before this struct can be used.
    assert!(size_of::<*const ContactShellTile>() == 8);

    // Operator §1.1 — mandatory const-context offset asserts (Zero-Trust).
    // Fail compilation immediately on FFI drift.  The C++ side has the
    // mirror static_assert in adjudicator.cuh.
    assert!(offset_of!(InterferometricAdjudicatorFfi, gear_override) == 100);
    assert!(offset_of!(InterferometricAdjudicatorFfi, force_prune_mask_ptr) == 104);
    assert!(offset_of!(InterferometricAdjudicatorFfi, potential_energy) == 112);
    assert!(offset_of!(InterferometricAdjudicatorFfi, d_dt) == 120);
    assert!(offset_of!(InterferometricAdjudicatorFfi, d_external_work) == 128);

    // M1.2.20.C-A — gasp handles + reserved tail.  Pinning these keeps
    // the FFI struct at exactly 256 B (two L1 sectors), and the C++
    // mirror in adjudicator.cuh has matching static_asserts.
    assert!(offset_of!(InterferometricAdjudicatorFfi, gasp_gain_eta) == 136);
    assert!(offset_of!(InterferometricAdjudicatorFfi, force_burst_step) == 140);
    assert!(offset_of!(InterferometricAdjudicatorFfi, momentum_violation_flag) == 144);
    // M1.2.26.RECOUPLE Part 2 — KL-native μ/σ at offsets 148/152.
    assert!(offset_of!(InterferometricAdjudicatorFfi, total_kl_mu) == 148);
    assert!(offset_of!(InterferometricAdjudicatorFfi, total_kl_sigma) == 152);
    // M1.2.26.RECOUPLE — adjudication_reason_flags relocated to 156.
    assert!(offset_of!(InterferometricAdjudicatorFfi, adjudication_reason_flags) == 156);
    assert!(offset_of!(InterferometricAdjudicatorFfi, _reserved_m1_2_26) == 160);
    assert!(size_of::<InterferometricAdjudicatorFfi>() == 256);
};

// ============================================================================
// FFI — extern "C" forward declarations for the C-side kernels
// ============================================================================

/// Public type alias for the CUDA error code at the FFI boundary.
/// Mirrors the C-side `cudaError_t` (an `int` enum).
pub type CudaError = i32;

/// `cudaSuccess` (= 0). Every public FFI call returns this on success.
pub const CUDA_SUCCESS: CudaError = 0;

#[cfg(feature = "gpu")]
pub(crate) mod ffi {
    use super::{CudaError, InterferometricAdjudicatorFfi};
    use std::ffi::c_void;

    extern "C" {
        /// Sentinel `0xADJ1`. Confirms the static archive containing
        /// the Adjudicator's CUDA TUs linked correctly and that the
        /// FFI ABI is round-tripping through the build pipeline.
        ///
        /// The Rust-side wrapper [`super::link_probe`] returns this
        /// value verbatim; tests assert the exact constant.
        pub fn prism_interferometric_adjudicator_link_probe() -> u32;

        /// Allocate one [`InterferometricAdjudicatorFfi`] from the
        /// F2 pool, zero-initialise it on the supplied stream, and
        /// return the device-side pointer.
        ///
        /// **Pointer-stable:** the returned pointer remains valid
        /// for the entire MD campaign. The caller MUST free via
        /// [`prism_interferometric_adjudicator_destroy`] before the
        /// pool is destroyed.
        ///
        /// `pool` is the `cudaMemPool_t` handle from
        /// [`crate::vram_pool::VramPool`].
        /// `stream` is a non-default `cudaStream_t`.
        pub fn prism_interferometric_adjudicator_create(
            pool: *mut c_void,
            stream: *mut c_void,
            out_ptr: *mut *mut InterferometricAdjudicatorFfi,
        ) -> CudaError;

        /// Free the F2-pool-backed adjudicator allocation. Must be
        /// called before [`crate::vram_pool::VramPool`] is dropped.
        pub fn prism_interferometric_adjudicator_destroy(
            ptr: *mut InterferometricAdjudicatorFfi,
            stream: *mut c_void,
        ) -> CudaError;

        /// Run the T13 SIMT 4-plane weighted KL Adjudicator kernel
        /// against the (relaxed[], perturbed[]) `ContactShellTile` arrays
        /// currently pointed-to by `adj`. Updates `current_divergence`,
        /// `adjudication_code`, `legacy_centroid_fallback` in place.
        ///
        /// **Launch geometry**: `<<<1, 64, 0, stream>>>`. Each thread
        /// processes one cluster (capped at 64). Threads with
        /// `id >= n_clusters` early-skip and contribute 0 to the
        /// block-level sum reduction.
        ///
        /// **Determinism**: integer outputs (`adjudication_code`) are
        /// BitExact across replays of the same input. KL accumulation
        /// uses IEEE-754 single-precision adds; ordering is fixed by
        /// the per-thread loop + tree-reduce.
        ///
        /// **Race-freedom**: thread 0's internal `__threadfence()`
        /// orders the `current_divergence` write before the
        /// `adjudication_code` write. The captured-graph topology
        /// orders this kernel before the F1 SWITCH node's read.
        ///
        /// Captured into the F1 conditional graph node's predicate
        /// evaluator — must NOT be invoked outside a captured graph
        /// in production paths.
        pub fn prism_interferometric_adjudicator_step(
            adj: *mut InterferometricAdjudicatorFfi,
            n_clusters: u32,
            stream: *mut c_void,
        ) -> CudaError;

        /// Update the running noise-floor estimates (mu, sigma) from
        /// the latest "Cool" frame's power spectrum. Captured into
        /// the graph; runs as `<<<1, 1, 0, stream>>>` — single
        /// thread, no atomic contention.
        pub fn prism_interferometric_adjudicator_update_noise_floor(
            adj: *mut InterferometricAdjudicatorFfi,
            cool_power_spectrum: *const f32, // 6 floats, l=0..5
            stream: *mut c_void,
        ) -> CudaError;

        // ─── T3 — ASC Boundary Repulsion Tensor ─────────────────────
        /// Per-atom F_i = α · Δ_AB · (x_i − X_c) atomicAdded into
        /// the existing `fused_engine.rs::d_forces` buffer
        /// (Anti-Greenfield § 2.1). Pointer-stability invariant on
        /// every device pointer.
        // M1.2.18-P3.2 — pe_components: per-atom PE accumulator (nullable).
        // When non-null, V_ASC = -½·α·Δ·‖x_i − X_c‖²·mask folded into
        // pe_components[i] so the SFA stability fuse sees the steering
        // potential as part of V_t.
        pub fn prism_asc_apply(
            adj: *const InterferometricAdjudicatorFfi,
            d_forces: *mut f32,
            d_atom_positions: *const f32,
            d_atom_in_cluster: *const u32,
            n_atoms: i32,
            steering_gain_alpha: f32,
            stream: *mut c_void,
            pe_components: *mut f64,
            // P7a — ASC application evidence, both nullable.
            // `d_asc_applied_atoms` counts atoms that received a non-zero
            // steering force; `d_asc_applied_sum_f` sums |F_ASC| over them.
            // Together they turn "is ASC engaged" from an inference into a
            // measurement the receipt can state.
            d_asc_applied_atoms: *mut u32,
            d_asc_applied_sum_f: *mut f32,
        ) -> CudaError;

        // ─── T4 — clock64 pipeline timing bookends ──────────────────
        /// FIRST kernel in the captured pipeline. Stamps adj->start_clock.
        pub fn prism_pipeline_clock_start(
            adj: *mut InterferometricAdjudicatorFfi,
            stream: *mut c_void,
        ) -> CudaError;
        /// LAST kernel in the captured pipeline. Stamps adj->stop_clock.
        pub fn prism_pipeline_clock_stop(
            adj: *mut InterferometricAdjudicatorFfi,
            stream: *mut c_void,
        ) -> CudaError;

        // ─── DAG-COND-WIRE — F1 SWITCH bridge ───────────────────────
        /// Returns a stable `*const u32` to the `adjudication_code`
        /// field at byte-offset 52. Pointer-stable for the campaign
        /// (G19 invariant). Use as the predicate-pointer argument to
        /// `cudaGraphConditionalHandleCreate` (memory-pointer model)
        /// or as input to [`prism_adj_set_conditional`] for the
        /// CUDA 12.4+ handle-based conditional-node model.
        pub fn prism_get_adjudication_code_devptr(
            adj: *const InterferometricAdjudicatorFfi,
        ) -> *const u32;

        /// Single-thread bridge kernel: reads the memory-resident
        /// `adj->adjudication_code` and forwards it via device-side
        /// `cudaGraphSetConditional(handle, code)` to the F1 SWITCH
        /// node. Place this kernel DOWNSTREAM of the adjudicator-step
        /// node and UPSTREAM of the conditional node, with explicit
        /// `cudaGraphAddDependencies` edges (operator mandate § 2.3).
        ///
        /// `handle` is `cudaGraphConditionalHandle` = `unsigned long long`
        /// (`driver_types.h:3229`), passed by value across FFI.
        ///
        /// Returns `cudaErrorNotSupported` (= 801) if the toolkit is
        /// pre-CUDA-12.4; in production (CUDA 13.x) this is always live.
        pub fn prism_adj_set_conditional(
            handle: u64, /* cudaGraphConditionalHandle */
            adj: *const InterferometricAdjudicatorFfi,
            stream: *mut c_void,
        ) -> CudaError;

        // ─── T7 — Noise-floor calibration writeback ─────────────────
        /// Two stream-ordered `cudaMemcpyAsync(HostToDevice)` operations
        /// that write the calibrated `μ[6]` / `σ[6]` priors into the
        /// adjudicator's noise-floor fields. Avoids recompilation;
        /// Claude-1's T7 calibration script captures the cold_hold
        /// equilibrium and writes back via this setter.
        ///
        /// `mu_host` / `sigma_host` are each pointers to 6 contiguous
        /// `f32` values. Pageable host memory may stage synchronously;
        /// pinned host memory (`cudaMallocHost`) is truly async.
        pub fn prism_adj_set_noise_floor_constants(
            adj: *mut InterferometricAdjudicatorFfi,
            mu_host: *const f32,
            sigma_host: *const f32,
            stream: *mut c_void,
        ) -> CudaError;
    }
}

/// Safe wrapper over the FFI link-probe. Returns `0xADJ1`
/// (= 0x_AD_J1 ≅ 0xAD_31_in_hex_from_glyph_proxy = 0xAD31).
///
/// In hex the sentinel is `0xADJ1` per spec, but `J` isn't a hex
/// digit; the canonical numeric value is `0xAD31` (a-d-3-1, with
/// the `J → 31` alphabet position). Tests pin the numeric value.
#[cfg(feature = "gpu")]
pub fn link_probe() -> u32 {
    unsafe { ffi::prism_interferometric_adjudicator_link_probe() }
}

/// Same numeric value the C-side kernel returns. Used by the link
/// probe and by integration tests to detect ABI drift.
pub const LINK_PROBE_SENTINEL: u32 = 0xAD31;

// ============================================================================
// T1 — UV-code / intensity bitwise extraction (CPU mirrors of the
//      __device__ helpers in cuda/adjudicator.cuh). Branchless. Tested
//      for round-trip with known producer-side encodings.
// ============================================================================

/// Bit shift of the 2-bit Quantum Identifier inside `intensity_packed`.
/// Mirrors `PRISM_QI_SHIFT` in `cuda/adjudicator.cuh`.
pub const QI_SHIFT: u32 = 30;

/// Mask for the lower-30-bit intensity payload.
/// Mirrors `PRISM_INTENSITY_PAYLOAD_MASK` in `cuda/adjudicator.cuh`.
pub const INTENSITY_PAYLOAD_MASK: u32 = 0x3FFF_FFFF;

/// μ_01² LUT canonical values — derived from `crate::config` extinction
/// coefficients per the Anti-Greenfield § 5 scavenging audit. Indexed
/// by the 2-bit UV code (0=260, 1=280, 2=305, 3=320 nm). MUST match
/// the C-side `c_mu_01_sq` initializer in `cuda/adjudicator.cu`.
pub const MU_01_SQ_LUT: [f32; 4] = [
    0.0343,    // 260 nm — PHE-dominant
    1.0000,    // 280 nm — TRP primary
    0.0621,    // 305 nm — TRP gaussian tail
    0.000_810, // 320 nm — TRP control / baseline
];

/// Extract the 2-bit UV code from `RichSpike::intensity_packed`.
#[inline]
pub fn extract_uv_code(intensity_packed: u32) -> u32 {
    (intensity_packed >> QI_SHIFT) & 0x3
}

/// Extract the lower-30-bit intensity payload as f32.
///
/// **Producer contract:** intensities must be encoded in `[0, 2.0)`
/// before packing. Values `≥ 2.0` lose bit 30 of the IEEE-754
/// exponent (biased exponent ≥ 128 ⇒ actual exponent ≥ 1) and
/// round-trip incorrectly. The producer side MUST clamp before
/// `intensity_packed = (uv_code << 30) | (intensity_bits & 0x3FFF_FFFF)`.
#[inline]
pub fn extract_intensity(intensity_packed: u32) -> f32 {
    f32::from_bits(intensity_packed & INTENSITY_PAYLOAD_MASK)
}

/// CPU mirror of `__device__ prism_compute_quantum_weight`. Returns
/// `Q_s = μ_01²(λ) · I` for the (UV-code, intensity) pair carried by
/// `intensity_packed`. **Bit-exact with the GPU kernel** for any input
/// that round-trips through the producer contract.
#[inline]
pub fn compute_quantum_weight(intensity_packed: u32) -> f32 {
    let uv_code = extract_uv_code(intensity_packed) as usize;
    MU_01_SQ_LUT[uv_code & 0x3] * extract_intensity(intensity_packed)
}

// ============================================================================
// T2 — CPU reference for the KL-divergence Adjudicator kernel.
//      Mirrors the algorithm in `cuda/adjudicator.cu::prism_inter
//      ferometric_adjudicator_step_kernel` so the dirty-tile synthetic
//      proof can run host-side without GPU execution.
// ============================================================================

/// Defensive epsilon — clamps zeros / negative finite values away
/// from the log singularity. Mirrors the C-side `epsilon` in
/// `prism_interferometric_adjudicator_step_kernel`.
pub const ADJUDICATOR_EPSILON: f32 = 1.0e-7;

/// CPU reference output: `(divergence, adjudication_code)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuAdjudicatorOutput {
    pub divergence: f32,
    pub adjudication_code: u32,
    pub violation_count: u32,
}

/// CPU mirror of the T2 KL-divergence kernel. Pure-logic, deterministic,
/// no I/O. The dirty-tile synthetic proof exercises all 64
/// combinations of edge-case inputs against this function.
///
/// Algorithm (matches GPU kernel byte-for-byte modulo
/// `lg2.approx.f32` rounding tolerance):
///
/// 1. For each band l = 0..6:
///    a. Detect dirty raw input (`!isfinite(p) || !isfinite(q)`),
///       increment `violation_count`, substitute `epsilon` for
///       non-finite raw values.
///    b. Defensive clamp: `p = max(p_raw, epsilon)`,
///       `q = max(q_raw, epsilon)`.
///    c. `ratio = p / q`; if `ratio.is_nan() || ratio <= 0.0` set
///       `log_ratio = 0.0`, else `log_ratio = ratio.log2()`.
///    d. Accumulate `total += p * log_ratio`.
/// 2. Final code:
///    - `violation_count > 0` ⇒ `Violation` (2)
///    - `total > μ_noise + 3σ` ⇒ `Construct` (1)
///    - else ⇒ `Prune` (0)
pub fn cpu_adjudicator_reference(
    p_spectrum: &[f32; 6],
    q_spectrum: &[f32; 6],
    noise_mu_l0: f32,
    noise_sigma_l0: f32,
) -> CpuAdjudicatorOutput {
    let mut total_kl_div = 0.0_f32;
    let mut violation_count: u32 = 0;
    let eps = ADJUDICATOR_EPSILON;

    for l in 0..6 {
        let mut p_raw = p_spectrum[l];
        let mut q_raw = q_spectrum[l];

        if !p_raw.is_finite() || !q_raw.is_finite() {
            violation_count += 1;
            if !p_raw.is_finite() {
                p_raw = eps;
            }
            if !q_raw.is_finite() {
                q_raw = eps;
            }
        }

        let p = p_raw.max(eps);
        let q = q_raw.max(eps);
        let ratio = p / q;

        // PTX-guard mirror: NaN OR ≤ 0 ⇒ log_ratio = 0.0.
        let log_ratio = if ratio.is_nan() || ratio <= 0.0 {
            0.0
        } else {
            ratio.log2()
        };

        total_kl_div += p * log_ratio;
    }

    let adjudication_code = if violation_count > 0 {
        2 // Violation
    } else {
        let threshold = noise_mu_l0 + 3.0 * noise_sigma_l0;
        if total_kl_div > threshold {
            1 // Construct
        } else {
            0 // Prune
        }
    };

    CpuAdjudicatorOutput {
        divergence: total_kl_div,
        adjudication_code,
        violation_count,
    }
}

// ============================================================================
// T3 — ASC Boundary Repulsion Tensor (CPU reference + Section-K helpers)
// ============================================================================

/// Per-atom force-vector contribution from the ASC kernel. Mirrors
/// the GPU kernel's per-atom output BEFORE the atomicAdd into the
/// shared `d_forces` buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AscForceContribution {
    pub fx: f32,
    pub fy: f32,
    pub fz: f32,
}

impl AscForceContribution {
    /// Squared L2 magnitude of the force vector.
    #[inline]
    pub fn magnitude_sq(&self) -> f32 {
        self.fx * self.fx + self.fy * self.fy + self.fz * self.fz
    }

    /// L2 magnitude of the force vector.
    #[inline]
    pub fn magnitude(&self) -> f32 {
        self.magnitude_sq().sqrt()
    }

    /// Dot product with another vector (useful for outward-direction tests).
    #[inline]
    pub fn dot(&self, other: [f32; 3]) -> f32 {
        self.fx * other[0] + self.fy * other[1] + self.fz * other[2]
    }
}

/// CPU reference for the T3 ASC kernel. Returns per-atom force
/// vectors WITHOUT performing atomicAdd. Pure-logic mirror of
/// `prism_asc_apply_kernel` in `cuda/adjudicator.cu`:
///
/// * `adjudication_code == 1` (Construct) and `in_cluster_mask[i]`
///   ⇒ `F_i = α · Δ_AB · (x_i − X_c)`, where `X_c` is the AABB midpoint.
/// * Otherwise ⇒ `F_i = (0, 0, 0)`.
///
/// Used by Section-K (Steering Coherence) tests + pre-integration
/// host-side validation.
pub fn cpu_asc_reference(
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    atom_positions: &[[f32; 3]],
    in_cluster_mask: &[bool],
    kl_divergence: f32,
    steering_gain_alpha: f32,
    adjudication_code: u32,
) -> Vec<AscForceContribution> {
    assert_eq!(
        atom_positions.len(),
        in_cluster_mask.len(),
        "atom_positions and in_cluster_mask must be parallel arrays",
    );

    let zero = AscForceContribution {
        fx: 0.0,
        fy: 0.0,
        fz: 0.0,
    };

    // Only Construct triggers the force apply (matches kernel's
    // `if (code != PRISM_ADJ_CONSTRUCT) return;` guard).
    if adjudication_code != 1 {
        return vec![zero; atom_positions.len()];
    }

    let xc = [
        0.5 * (aabb_min[0] + aabb_max[0]),
        0.5 * (aabb_min[1] + aabb_max[1]),
        0.5 * (aabb_min[2] + aabb_max[2]),
    ];
    let scale_base = steering_gain_alpha * kl_divergence;

    atom_positions
        .iter()
        .zip(in_cluster_mask.iter())
        .map(|(xi, &in_cluster)| {
            let mask = if in_cluster { 1.0 } else { 0.0 };
            let scale = scale_base * mask;
            AscForceContribution {
                fx: scale * (xi[0] - xc[0]),
                fy: scale * (xi[1] - xc[1]),
                fz: scale * (xi[2] - xc[2]),
            }
        })
        .collect()
}

// ============================================================================
// T4 — clock64 cycles ↔ nanoseconds conversion (sm_120 boost = 2.977 GHz)
// ============================================================================

/// Blackwell sm_120 boost clock in GHz, per operator tech spec
/// (2.977 GHz / 2977 MHz). Used to convert `clock64()` cycle counts
/// to nanoseconds for the < 10 μs CSR-H gate.
pub const BLACKWELL_BOOST_GHZ: f64 = 2.977;

/// Convert SM cycle count to nanoseconds at Blackwell boost.
///
/// `1 ns ≈ 2.977 cycles` ⇒ `cycles ≈ ns × 2.977` ⇒
/// `ns ≈ cycles / 2.977`.
#[inline]
pub fn cycles_to_ns(cycles: u64) -> f64 {
    (cycles as f64) / BLACKWELL_BOOST_GHZ
}

/// Pipeline-elapsed time, computed from the `start_clock` /
/// `stop_clock` bookend timestamps written by
/// `prism_pipeline_clock_start` / `_stop` kernels.
///
/// Returns `(cycles, ns)`.
///
/// **CSR-H gate**: `ns < 10_000.0` (i.e., the full
/// SO(3) → KL-Div → ASC pipeline must complete in < 10 μs).
pub fn pipeline_elapsed(adj: &InterferometricAdjudicatorFfi) -> (u64, f64) {
    let cycles = adj.stop_clock.saturating_sub(adj.start_clock);
    (cycles, cycles_to_ns(cycles))
}

// ============================================================================
// T7 — Calibrated noise-floor constants (LOCKED 2026-04-30)
// ============================================================================
//
// Operator-published μ/σ priors for the 3-σ Adjudicator threshold.
// Measured by Claude-1's stripped canonical calibration on 4LPK
// during a 500-frame cold_hold equilibrium. The thermodynamic
// baseline is locked here as compile-time constants so every MD
// campaign init — across every one of the 18 643 targets in the
// Blackwell Convergence main run — applies the same priors via
// `apply_t7_calibration()` after `prism_interferometric_adjudicator_create`.
//
// **Source of truth provenance:**
//   - Target:      4LPK (KRAS) cold_hold phase
//   - Frames:      500 (Phase-0 calibration)
//   - Methodology: Σ KL contribution per SH band l ∈ 0..5, taken
//                  in thermal equilibrium (no UV perturbation)
//   - Date locked: 2026-04-30 (per Architect's published values)
//   - Locked by:   Lane 2 (Claude-2) via this commit
//   - Verified by: Lane 1 (Claude-1) calibration script, prior commit
//
// Drift policy: re-calibration produces NEW constants under a NEW
// public name (e.g., `T7_CALIBRATED_MU_V2`). Editing these literals
// in place is FORBIDDEN — every change must be a code commit so the
// audit trail captures who changed what when. The regression test
// `t7_constants_match_operator_published_values` enforces byte-exact
// stability of the locked values.

/// Per-SH-band running mean of the thermal noise-floor's Σ KL
/// contribution. Index `l` corresponds to spherical-harmonic band
/// `l = 0..5`. Operator-published, locked 2026-04-30. See module
/// header for full provenance.
pub const T7_CALIBRATED_MU: [f32; 6] = [
    0.8052561253, // l=0
    0.0040383553, // l=1
    0.0703344136, // l=2
    0.0538048399, // l=3
    0.0396647932, // l=4
    0.0269014686, // l=5
];

/// Per-SH-band running stddev of the thermal noise-floor's Σ KL
/// contribution. Pairs with `T7_CALIBRATED_MU` to define the
/// 3-σ Adjudicator threshold `μ + 3σ` per band. Operator-published,
/// locked 2026-04-30.
pub const T7_CALIBRATED_SIGMA: [f32; 6] = [
    0.1482125481, // l=0
    0.0090773341, // l=1
    0.0805278777, // l=2
    0.0222988033, // l=3
    0.0565869628, // l=4
    0.0099504697, // l=5
];

/// Convenience helper: write the locked T7 constants into the
/// adjudicator's device state in a single call. Equivalent to:
///
/// ```ignore
/// set_noise_floor_constants(adj, &T7_CALIBRATED_MU, &T7_CALIBRATED_SIGMA, stream)
/// ```
///
/// Recommended invocation site (Claude-1's `nhs_rt_full.rs` runtime):
///
/// ```ignore
/// // Inside MD-campaign init, after adjudicator_create + before any
/// // captured-graph launch:
/// let rc = unsafe { apply_t7_calibration(adj_devptr, stream_raw) };
/// assert_eq!(rc, CUDA_SUCCESS, "T7 calibration apply failed: {}", rc);
/// stream.synchronize()?;
/// ```
///
/// **Safety contract**: identical to [`set_noise_floor_constants`].
#[cfg(feature = "gpu")]
#[inline]
pub unsafe fn apply_t7_calibration(
    adj: *mut InterferometricAdjudicatorFfi,
    stream: *mut std::ffi::c_void,
) -> CudaError {
    set_noise_floor_constants(adj, &T7_CALIBRATED_MU, &T7_CALIBRATED_SIGMA, stream)
}

// ============================================================================
// Wave 0 / Task #68 — KL-DIVERGENCE NOISE-FLOOR BOOTSTRAP
// ============================================================================
//
// The Adjudicator step kernel's threshold formula is
//   threshold = adj->noise_floor_mu[0] + 3.0f * adj->noise_floor_sigma[0]
// and the test it gates is `fabsf(total_kl) > threshold` (Wave 0 fix
// 2026-05-02, prior code compared signed `total_kl` against the same
// threshold and silently dropped half the divergence half-plane).
//
// `total_kl` is the 4-plane WEIGHTED KL DIVERGENCE
// `Σ_planes ω_p · Σ_l p_l · log2(p_l/q_l)`.  It is dimensionless and
// — under cold-hold (relaxed≈perturbed manifolds) — should sit at
// magnitudes ≪ 1.  Warm-phase divergence events run several orders
// of magnitude larger (the legacy CPU PCMI/SURP loop sees Σ KL
// bursts in the 10² range on 7C8R).
//
// `T7_CALIBRATED_MU` / `T7_CALIBRATED_SIGMA` (above) are the
// **C_l power-spectrum** statistics from cold-hold sampling.  Plugging
// those into `adj->noise_floor_mu[0..6]` (μ[0]≈0.805) gave a threshold
// of ~1.249 — comparing apples to oranges.  The 15k 7C8R campaign
// (2026-05-02) exposed this: 35k steps, 0 GhostTileFrame records.
//
// The KL bootstrap below seeds `adj->noise_floor_mu[0]` and
// `adj->noise_floor_sigma[0]` with KL-magnitude priors small enough
// that the very first warm-phase divergence event lifts above the
// threshold, while Dynamic T7 (`dynamic_t7.cu`) takes over at
// `PRISM_DYNT7_N_MIN = 100` samples and replaces the bootstrap with
// the substrate's measured cold-hold KL μ/σ via stream-ordered
// `atomicMin`/`atomicMax`-free in-place writes.
//
// Why these specific numbers
//   μ_kl_boot      = 0.0    (cold-hold KL is centred near zero in
//                            both directions — the relaxed and
//                            perturbed manifolds are statistically
//                            identical in equilibrium).
//   σ_kl_boot      = 1.0e-3 (2-σ KL noise envelope from the 4LPK
//                            cold-hold profile, conservatively
//                            doubled to avoid first-frame
//                            false-Construct on numerical jitter
//                            before Dynamic T7 has 100 samples).
//   threshold_boot = μ + 3σ = 3.0e-3
//
// Bands [1..5] are written but unused by the kernel (it only reads
// index 0).  We seed them with the same values for symmetry; future
// per-band threshold support can repopulate without ABI churn.
//
// Lock policy: same as `T7_CALIBRATED_*` — re-calibration produces
// new constants under a new public name; in-place mutation
// FORBIDDEN.

/// **Wave 0 / Task #68** — KL-units μ bootstrap for `noise_floor_mu`.
///
/// Per-band (only index 0 is read by the step kernel).  All bands
/// share the same value here; per-band differentiation is a future
/// enhancement once Dynamic T7 captures per-band KL samples.
pub const T7_KL_BOOTSTRAP_MU: [f32; 6] = [0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32];

/// **Wave 0 / Task #68** — KL-units σ bootstrap for `noise_floor_sigma`.
///
/// 1.0e-3 → bootstrap threshold of 3.0e-3.  Dynamic T7 overwrites
/// index 0 with the measured substrate value once it has ≥100
/// `current_divergence` samples; bands [1..5] retain this value.
pub const T7_KL_BOOTSTRAP_SIGMA: [f32; 6] = [
    1.0e-3_f32, 1.0e-3_f32, 1.0e-3_f32, 1.0e-3_f32, 1.0e-3_f32, 1.0e-3_f32,
];

/// **Wave 0 / Task #68** — apply the KL-units bootstrap into the
/// adjudicator's device state.  This REPLACES `apply_t7_calibration`
/// for the captured-pipeline init path: the C_l-magnitude priors
/// (`T7_CALIBRATED_MU/SIGMA`) yielded a threshold of ~1.249 — the
/// wrong order of magnitude vs. the divergence the kernel is
/// actually computing.  See module-level comment above for the
/// numerics rationale.
///
/// **Safety contract**: identical to [`set_noise_floor_constants`].
#[cfg(feature = "gpu")]
#[inline]
pub unsafe fn apply_t7_kl_calibration(
    adj: *mut InterferometricAdjudicatorFfi,
    stream: *mut std::ffi::c_void,
) -> CudaError {
    set_noise_floor_constants(adj, &T7_KL_BOOTSTRAP_MU, &T7_KL_BOOTSTRAP_SIGMA, stream)
}

// ============================================================================
// T7 — Noise-floor calibration setter (safe Rust wrapper)
// ============================================================================

/// Safe wrapper over `prism_adj_set_noise_floor_constants` FFI.
///
/// Writes the calibrated `μ[6]` and `σ[6]` priors into the device-side
/// adjudicator state via two stream-ordered `cudaMemcpyAsync(H2D)`
/// operations. Returns `0` on success, otherwise the CUDA error code.
///
/// **Lifecycle**: call ONCE per MD campaign after T7 cold_hold
/// calibration completes, BEFORE the first captured-graph launch.
/// May also be re-called between launches to recalibrate without
/// rebuilding the captured graph (the adjudicator pointer is stable
/// per the F2-pool `ReleaseThreshold = UINT64_MAX` invariant).
///
/// **Safety contract:**
///   - `adj` must be a valid F2-pool-allocated
///     [`InterferometricAdjudicatorFfi`] device pointer whose lifetime
///     spans every captured-graph replay that follows.
///   - `stream` must be a valid `cudaStream_t`.
///   - For truly async writes, `mu` and `sigma` should reside in
///     pinned host memory (`cudaMallocHost`); pageable host memory
///     may force a synchronous staging copy.
///   - Caller is responsible for any sync that must precede a
///     subsequent device read of the noise-floor fields (typically
///     a `stream.synchronize()` or downstream graph-edge dependency).
///
/// Example calibration flow (Claude-1 T7 calibration script):
///
/// ```ignore
/// // 1. Run 500-frame cold_hold; collect per-band Σ KL contributions.
/// let (mu_calibrated, sigma_calibrated) = run_t7_calibration(...)?;
/// // 2. Write back to the adjudicator's device state.
/// let rc = unsafe {
///     set_noise_floor_constants(
///         adj_dev_ptr,
///         &mu_calibrated,
///         &sigma_calibrated,
///         stream.as_raw() as *mut std::ffi::c_void,
///     )
/// };
/// assert_eq!(rc, CUDA_SUCCESS, "T7 calibration writeback failed");
/// stream.synchronize()?;
/// ```
#[cfg(feature = "gpu")]
pub unsafe fn set_noise_floor_constants(
    adj: *mut InterferometricAdjudicatorFfi,
    mu: &[f32; 6],
    sigma: &[f32; 6],
    stream: *mut std::ffi::c_void,
) -> CudaError {
    ffi::prism_adj_set_noise_floor_constants(adj, mu.as_ptr(), sigma.as_ptr(), stream)
}

// ============================================================================
// DAG-COND-WIRE — F1 SWITCH predicate-handle accessor (G19 surface)
// ============================================================================

/// Byte offset of the `adjudication_code` field within
/// [`InterferometricAdjudicatorFfi`]. Pinned at 52 by the
/// CSR-C invariant (verified by `tests::ffi_field_offsets_match_csr_c_table`).
///
/// Used by:
///   - The F1 SWITCH wiring code that creates a
///     `cudaGraphConditionalHandle` bound to this address.
///   - The G19 sub-byte address-stability test below.
pub const ADJUDICATION_CODE_OFFSET: usize = 52;

/// Returns a stable raw pointer to the `adjudication_code` field
/// suitable for the F1 SWITCH predicate handle.
///
/// **Safety contract:**
///   - `adj` must be a valid pointer to a properly-aligned
///     [`InterferometricAdjudicatorFfi`] whose lifetime spans the
///     entire captured-graph WHILE region.
///   - The caller MUST NOT free the underlying allocation while any
///     graph launch references the predicate.
///
/// The returned pointer is:
///   - **4-byte aligned** (sufficient for atomic LDG.E.32 read).
///   - Within the same 128-byte L1 sector as the struct head
///     (no split-load penalty on the F1 hardware-scheduler read).
///   - **Pointer-stable** for the campaign per the F2 pool's
///     `ReleaseThreshold = UINT64_MAX` guarantee.
///
/// Usage with CUDA 12.4+ `cudaGraphConditionalHandleCreate`:
///
/// ```ignore
/// let predicate = adjudication_code_devptr(adj);
/// // Pass predicate as the device-side address bound to the handle,
/// // OR launch prism_adj_set_conditional to forward via cudaGraphSetConditional.
/// ```
#[inline]
pub unsafe fn adjudication_code_devptr(adj: *const InterferometricAdjudicatorFfi) -> *const u32 {
    // Use addr_of! to compute the field address without dereferencing
    // (place expression). The cast back via byte_offset is equivalent
    // and confirms the offset matches the CSR-C invariant.
    std::ptr::addr_of!((*adj).adjudication_code)
}

// ============================================================================
// Pointer-stability lifecycle contract (operator mandate § 4)
// ============================================================================

/// **Lifecycle invariant** for every captured-graph WHILE iteration:
///
/// 1. Call `prism_interferometric_adjudicator_create` ONCE per
///    stream at MD-campaign init — **OUTSIDE** the
///    `cudaStreamBeginCapture` block.
/// 2. The returned `InterferometricAdjudicatorFfi*` pointer remains
///    valid for the entire campaign. The F2 mempool's
///    `ReleaseThreshold = UINT64_MAX` keeps it stable across
///    every replay.
/// 3. Inside the captured WHILE region: NEVER call create/destroy.
///    NEVER cudaMalloc/cudaFree. Doing so triggers an illegal memory
///    access in the Blackwell hardware scheduler on first replay
///    and (per operator warning) "the workstation will require a
///    hard reset."
/// 4. Update `relaxed_manifold_ptr` / `perturbed_manifold_ptr` via
///    `cudaGraphKernelNodeSetParams` (the parameter-update API
///    that's safe to call between captures).
pub const POINTER_STABILITY_CONTRACT: &str =
    "create() OUTSIDE capture; destroy() OUTSIDE capture; \
     no malloc/free inside the captured WHILE region.";

// ============================================================================
// Tests — runtime layout pins (CSR-C explicit byte-offsets)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    /// `offset_of!` polyfill — `std::mem::offset_of!` stabilised in
    /// Rust 1.77 (March 2024). If the workspace toolchain is older,
    /// swap this for the `memoffset` crate without touching call
    /// sites.
    macro_rules! offset_of {
        ($t:ty, $f:tt) => {{
            // Construct a typed null pointer, project to the field,
            // and read its byte offset via wrapping pointer arithmetic.
            // SAFETY: never dereferenced; only the pointer offset is
            // taken.
            let base: *const $t = std::ptr::null();
            unsafe {
                (std::ptr::addr_of!((*base).$f) as *const u8).offset_from(base as *const u8)
                    as usize
            }
        }};
    }

    #[test]
    fn ffi_struct_size_is_256_bytes() {
        // M1.2.18.5 — explicit 256-byte lock (two Blackwell L1 sectors).
        // The first sector (0..128) holds the SO(3) outputs +
        // legacy_centroid_fallback + force_prune_mask + d_potential_energy +
        // d_dt.  The second (128..256) holds d_external_work pointer +
        // gear_override + tail padding.  Drift here ⇒ breaks both Rust-side
        // offset_of asserts and C++-side static_assert(sizeof == 256).
        assert_eq!(
            size_of::<InterferometricAdjudicatorFfi>(),
            256,
            "InterferometricAdjudicatorFfi MUST be 256 bytes (M1.2.18.5 \
             two-sector lock). Drift means one of: missing field, missing \
             padding, or compiler-inserted alignment changed.  Cross-check \
             the C-side static_assert in adjudicator.cuh."
        );
    }

    #[test]
    fn ffi_struct_alignment_is_128_bytes() {
        assert_eq!(
            align_of::<InterferometricAdjudicatorFfi>(),
            128,
            "InterferometricAdjudicatorFfi MUST be 128-byte aligned (Blackwell L1 sector \
             boundary). Failure here means one of two things: the #[repr(C, align(128))] \
             attribute was removed, or the toolchain doesn't honour it on this target."
        );
    }

    #[test]
    fn ffi_field_offsets_match_csr_c_table() {
        // Explicit byte-offsets per CSR-section C requirement.
        // Layout drift here = silent ABI break against the C-side
        // mirror in adjudicator.cuh (its static_asserts pin the same
        // M1.2.18.5 layout: gear_override @ 136, d_external_work
        // (pointer) @ 128, total size 256.
        // values).
        assert_eq!(offset_of!(InterferometricAdjudicatorFfi, noise_floor_mu), 0);
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, noise_floor_sigma),
            24
        );
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, current_divergence),
            48
        );
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, adjudication_code),
            52
        );
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, relaxed_manifold_ptr),
            56
        );
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, perturbed_manifold_ptr),
            64
        );
        assert_eq!(offset_of!(InterferometricAdjudicatorFfi, start_clock), 72);
        assert_eq!(offset_of!(InterferometricAdjudicatorFfi, stop_clock), 80);
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, legacy_centroid_fallback),
            88
        );
        // Emergency Rectification — gear_override @ 100 (B.3.2 home).
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, gear_override),
            100
        );
        // Renamed: force_prune_mask → force_prune_mask_ptr (G28 SISR).
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, force_prune_mask_ptr),
            104
        );
        // Renamed: d_potential_energy → potential_energy (f64 VALUE).
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, potential_energy),
            112
        );
        assert_eq!(offset_of!(InterferometricAdjudicatorFfi, d_dt), 120);
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, d_external_work),
            128
        );
        // M1.2.20.C-A — gasp handles + 256-B explicit total.
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, gasp_gain_eta),
            136
        );
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, force_burst_step),
            140
        );
        // M1.2.20.C-B — Momentum Guard flag carved from _reserved.
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, momentum_violation_flag),
            144
        );
        // M1.2.20.C-G / T24 — adjudication_reason_flags (replaces lqi_flags).
        // M1.2.26.RECOUPLE — KL-native μ/σ at 148/152;
        // adjudication_reason_flags relocated to 156; _reserved tail at 160.
        assert_eq!(offset_of!(InterferometricAdjudicatorFfi, total_kl_mu), 148);
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, total_kl_sigma),
            152
        );
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, adjudication_reason_flags),
            156
        );
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, _reserved_m1_2_26),
            160
        );
        assert_eq!(std::mem::size_of::<InterferometricAdjudicatorFfi>(), 256);
    }

    #[test]
    fn zero_constructor_yields_all_zero_state() {
        let z = InterferometricAdjudicatorFfi::zero();
        assert_eq!(z.noise_floor_mu, [0.0; 6]);
        assert_eq!(z.noise_floor_sigma, [0.0; 6]);
        assert_eq!(z.current_divergence, 0.0);
        assert_eq!(z.adjudication_code, 0);
        assert!(z.relaxed_manifold_ptr.is_null());
        assert!(z.perturbed_manifold_ptr.is_null());
        assert_eq!(z.start_clock, 0);
        assert_eq!(z.stop_clock, 0);
        assert_eq!(z.legacy_centroid_fallback, [0.0; 3]);
        assert_eq!(
            z.gear_override,
            InterferometricAdjudicatorFfi::GEAR_OVERRIDE_AUTO
        );
        assert!(z.force_prune_mask_ptr.is_null());
        // potential_energy is an f64 value (not a ptr); zero by default.
        assert_eq!(z.potential_energy, 0.0_f64);
        assert!(z.d_dt.is_null());
        // d_external_work is a *mut f64 pointer; null by default.
        // Captured pipeline build wires it pre-capture and zero-host
        // guard rejects null at capture time (Operator §3 Zero-Host Guard).
        assert!(z.d_external_work.is_null());
    }

    #[test]
    fn adjudication_code_uses_pre_rank_canonical_enum() {
        // Anti-Greenfield § 5: the F1 SWITCH selector reuses the
        // existing AdjudicationCode enum from RECT-2 (commit
        // `0993bf9b`) — Prune=0, Construct=1, Violation=2 — rather
        // than defining parallel constants. Test pins the integer
        // mapping so that an FFI drift on EITHER side surfaces here.
        use crate::pre_rank::AdjudicationCode;
        assert_eq!(AdjudicationCode::Prune as u32, 0);
        assert_eq!(AdjudicationCode::Construct as u32, 1);
        assert_eq!(AdjudicationCode::Violation as u32, 2);

        // Round-trip through the field. The Adjudicator kernel
        // writes one of {0, 1, 2} into adjudication_code; host
        // decodes via from_raw.
        let mut adj = InterferometricAdjudicatorFfi::zero();
        adj.adjudication_code = 1;
        assert_eq!(
            AdjudicationCode::from_raw(adj.adjudication_code),
            Some(AdjudicationCode::Construct),
        );
        adj.adjudication_code = 2;
        assert_eq!(
            AdjudicationCode::from_raw(adj.adjudication_code),
            Some(AdjudicationCode::Violation),
        );
    }

    #[test]
    fn link_probe_sentinel_is_pinned() {
        // A drift here means the FFI ABI changed without a C-side
        // bump.  See `prism_interferometric_adjudicator_link_probe`.
        assert_eq!(LINK_PROBE_SENTINEL, 0xAD31);
    }

    // ─── T1 bitwise extraction round-trip ──────────────────────────────

    #[test]
    fn uv_code_extraction_is_branchless_and_correct() {
        // Bit pattern: top 2 bits = uv_code, lower 30 = intensity bits.
        // Encode intensity 1.0 (f32 bits 0x3F800000) with uv_code=2 (305 nm).
        let intensity_1_bits = 1.0_f32.to_bits(); // 0x3F80_0000
                                                  // intensity 1.0 fits in lower 30 bits because its IEEE-754 bits
                                                  // 0x3F800000 already have bits 30-31 == 00 (sign=0, exp top=0).
        let packed = (2_u32 << QI_SHIFT) | (intensity_1_bits & INTENSITY_PAYLOAD_MASK);
        assert_eq!(extract_uv_code(packed), 2);
        assert_eq!(extract_intensity(packed), 1.0);
    }

    #[test]
    fn quantum_weight_matches_lut() {
        // Pack intensity 1.0 with uv_code=1 (280 nm — TRP primary).
        // Q_s should equal MU_01_SQ_LUT[1] * 1.0 = 1.0000.
        let intensity_1_bits = 1.0_f32.to_bits();
        let packed_280 = (1_u32 << QI_SHIFT) | (intensity_1_bits & INTENSITY_PAYLOAD_MASK);
        let weight_280 = compute_quantum_weight(packed_280);
        assert!((weight_280 - 1.0).abs() < 1e-6);

        // Pack intensity 1.0 with uv_code=3 (320 nm — control). Should
        // be the smallest weight.
        let packed_320 = (3_u32 << QI_SHIFT) | (intensity_1_bits & INTENSITY_PAYLOAD_MASK);
        let weight_320 = compute_quantum_weight(packed_320);
        assert!(weight_320 < weight_280);
        assert!((weight_320 - 0.000_810).abs() < 1e-6);

        // Wavelength ordering invariant: 280 > 305 > 260 > 320.
        let packed_260 = (0_u32 << QI_SHIFT) | (intensity_1_bits & INTENSITY_PAYLOAD_MASK);
        let packed_305 = (2_u32 << QI_SHIFT) | (intensity_1_bits & INTENSITY_PAYLOAD_MASK);
        let w_260 = compute_quantum_weight(packed_260);
        let w_305 = compute_quantum_weight(packed_305);
        assert!(weight_280 > w_305, "TRP λ_max should outweigh tail");
        assert!(w_305 > w_260, "TRP tail at 305 should outweigh PHE at 260");
        assert!(
            w_260 > weight_320,
            "PHE at 260 should outweigh control at 320"
        );
    }

    // ─── T2 dirty-tile 64-combination Total Function proof ──────────────

    /// The 8 edge-case values per the directive's § 3.3 test matrix.
    const EDGE_CASES: [f32; 8] = [
        0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0e-30,
        1.0e+30,
        1.0,
        -1.0,
    ];

    #[test]
    fn dirty_tile_64_combination_total_function_proof() {
        // Operator directive § 3.3: for every (P, Q) ∈ EDGE_CASES²,
        // the Adjudicator MUST produce adjudication_code ∈ {0, 1, 2}.
        // The Total Function guarantee is what makes the F1 SWITCH
        // routing safe — no input pattern can produce an out-of-range
        // code that would crash the hardware scheduler.
        let mut violation_count_combos = 0;
        let mut prune_combos = 0;
        let mut construct_combos = 0;

        for &p_val in EDGE_CASES.iter() {
            for &q_val in EDGE_CASES.iter() {
                // All 6 bands populated with the same edge-case
                // value — worst case for the violation accumulator
                // (every band counts).
                let p_spec = [p_val; 6];
                let q_spec = [q_val; 6];

                let out = cpu_adjudicator_reference(&p_spec, &q_spec, 0.0, 1.0);

                // Total Function: code MUST be in {0, 1, 2}.
                assert!(
                    out.adjudication_code <= 2,
                    "Total-Function violation: P={:?} Q={:?} → code={}",
                    p_val,
                    q_val,
                    out.adjudication_code,
                );

                match out.adjudication_code {
                    0 => prune_combos += 1,
                    1 => construct_combos += 1,
                    2 => violation_count_combos += 1,
                    _ => unreachable!("guarded by assert above"),
                }
            }
        }

        // Sanity bounds — at least one combination of each class.
        // Specifically: combinations involving NaN / Inf / -Inf must
        // produce Violation (5 dirty values × 8 partners + 8 partners
        // × 5 dirty values − 5×5 dup = 39 dirty-touching combos).
        // Clean ∩ small or equal pairs go to Prune.
        // Clean ∩ large-asymmetry pairs go to Construct.
        assert_eq!(prune_combos + construct_combos + violation_count_combos, 64);
        assert!(
            violation_count_combos >= 39,
            "expected ≥ 39 Violation combos (any NaN/Inf/-Inf touch); got {}",
            violation_count_combos
        );
        assert!(prune_combos > 0, "expected ≥ 1 Prune combo");
        assert!(construct_combos > 0, "expected ≥ 1 Construct combo");
    }

    #[test]
    fn dirty_tile_specific_extreme_inputs_route_correctly() {
        // CSR-section D.2 Table — operator-mandated explicit cases.
        // Each row: (description, p_val, q_val, expected_code).
        let cases: &[(&str, f32, f32, u32)] = &[
            ("P=Inf  Q=NaN  → Violation", f32::INFINITY, f32::NAN, 2),
            ("P=NaN  Q=Inf  → Violation", f32::NAN, f32::INFINITY, 2),
            ("P=-Inf Q=1.0  → Violation", f32::NEG_INFINITY, 1.0, 2),
            ("P=NaN  Q=NaN  → Violation", f32::NAN, f32::NAN, 2),
            ("P=0.0  Q=0.0  → Prune (clean)", 0.0, 0.0, 0),
            ("P=1.0  Q=1.0  → Prune (equal)", 1.0, 1.0, 0),
            ("P=-1.0 Q=1.0  → Prune (clamped)", -1.0, 1.0, 0),
            ("P=1e+30 Q=1.0 → Construct", 1.0e+30, 1.0, 1),
        ];
        for &(desc, p, q, expected_code) in cases {
            let out = cpu_adjudicator_reference(&[p; 6], &[q; 6], 0.0, 1.0);
            assert_eq!(
                out.adjudication_code, expected_code,
                "case `{}`: got code={} (divergence={})",
                desc, out.adjudication_code, out.divergence,
            );
        }
    }

    #[test]
    fn deterministic_across_100_replays_per_combination() {
        // The Adjudicator is deterministic — feeding the same
        // (P, Q) pair 100 times MUST produce the same divergence
        // and the same adjudication_code every time.  This is
        // CSR-section D's gate.
        for &p_val in EDGE_CASES.iter() {
            for &q_val in EDGE_CASES.iter() {
                let p_spec = [p_val; 6];
                let q_spec = [q_val; 6];
                let first = cpu_adjudicator_reference(&p_spec, &q_spec, 0.0, 1.0);
                for replay in 0..100 {
                    let again = cpu_adjudicator_reference(&p_spec, &q_spec, 0.0, 1.0);
                    assert_eq!(
                        first.adjudication_code, again.adjudication_code,
                        "P={:?} Q={:?} replay {} drifted code: {} vs {}",
                        p_val, q_val, replay, first.adjudication_code, again.adjudication_code,
                    );
                    // Divergence: NaN ≠ NaN per IEEE-754, so compare
                    // bit patterns when either is non-finite.
                    if first.divergence.is_finite() && again.divergence.is_finite() {
                        assert_eq!(
                            first.divergence, again.divergence,
                            "P={:?} Q={:?} replay {} divergence drift",
                            p_val, q_val, replay,
                        );
                    } else {
                        assert_eq!(
                            first.divergence.to_bits(),
                            again.divergence.to_bits(),
                            "P={:?} Q={:?} replay {} non-finite divergence drift",
                            p_val,
                            q_val,
                            replay,
                        );
                    }
                }
            }
        }
    }

    // ─── T2 clean-input sanity checks ───────────────────────────────────

    #[test]
    fn equal_distributions_yield_zero_divergence() {
        // KL(P || P) = 0 for any non-degenerate P. The CPU reference
        // should produce exactly 0 (both sides clamp to the same
        // epsilon-padded value, ratio = 1, log = 0).
        let p_spec = [1.0; 6];
        let out = cpu_adjudicator_reference(&p_spec, &p_spec, 0.0, 1.0);
        assert_eq!(out.divergence, 0.0);
        assert_eq!(out.violation_count, 0);
        assert_eq!(out.adjudication_code, 0); // Prune (below threshold)
    }

    #[test]
    fn high_asymmetry_yields_construct_above_threshold() {
        // P >> Q in every band ⇒ KL > 0 ⇒ Construct above 3-σ.
        let p_spec = [10.0_f32; 6];
        let q_spec = [0.001_f32; 6];
        let out = cpu_adjudicator_reference(&p_spec, &q_spec, 0.0, 1.0);
        assert!(
            out.divergence > 3.0,
            "expected KL > threshold (3.0), got {}",
            out.divergence
        );
        assert_eq!(out.violation_count, 0);
        assert_eq!(out.adjudication_code, 1); // Construct
    }

    // ─── Section K — ASC Steering Coherence (per directive § 5) ─────────

    /// **Section K.1 — vector plot.** Eight atoms at the eight corners
    /// of a unit cube centred on origin. AABB centroid is origin. The
    /// ASC kernel must produce a force vector for each atom that
    /// points OUTWARD — i.e., `dot((atom − X_c), F) > 0` for every
    /// atom. This is the "Gasp" invariant: the simulation is being
    /// driven to expand the cluster envelope, not contract it.
    #[test]
    fn asc_section_k1_eight_octant_outward_forces() {
        let aabb_min = [-1.0_f32, -1.0, -1.0];
        let aabb_max = [1.0_f32, 1.0, 1.0];
        let atoms: Vec<[f32; 3]> = vec![
            [1.0, 1.0, 1.0],    // octant +++
            [-1.0, 1.0, 1.0],   // octant -++
            [1.0, -1.0, 1.0],   // octant +-+
            [-1.0, -1.0, 1.0],  // octant --+
            [1.0, 1.0, -1.0],   // octant ++-
            [-1.0, 1.0, -1.0],  // octant -+-
            [1.0, -1.0, -1.0],  // octant +--
            [-1.0, -1.0, -1.0], // octant ---
        ];
        let mask = vec![true; 8];
        let kl = 1.0_f32;
        let alpha = 1.0_f32;
        let forces = cpu_asc_reference(aabb_min, aabb_max, &atoms, &mask, kl, alpha, 1);

        // Coordinate list — every (atom, force) pair has positive
        // outward-direction dot product. Print on failure for the
        // CSR-K coordinate table.
        for (i, (atom, f)) in atoms.iter().zip(forces.iter()).enumerate() {
            let outward_dot = f.dot(*atom); // X_c is origin so atom_vec == atom
            assert!(
                outward_dot > 0.0,
                "octant {} NOT outward: atom={:?} force=({:.3},{:.3},{:.3}) dot={:.3}",
                i,
                atom,
                f.fx,
                f.fy,
                f.fz,
                outward_dot,
            );
            // Stronger: for this symmetric setup, force vector should
            // be exactly proportional to the atom vector.
            assert_eq!(f.fx.signum(), atom[0].signum(), "octant {} fx sign", i);
            assert_eq!(f.fy.signum(), atom[1].signum(), "octant {} fy sign", i);
            assert_eq!(f.fz.signum(), atom[2].signum(), "octant {} fz sign", i);
        }
    }

    /// **Section K.2 — linear scaling with KL divergence.** Force
    /// magnitude must scale linearly with `Δ_AB` for the same atom
    /// configuration. This is the "magnitude modulation" invariant
    /// (operator directive § 2.1: spring constant k ∝ KL-Divergence).
    #[test]
    fn asc_section_k2_force_magnitude_linear_in_kl_divergence() {
        let aabb_min = [-1.0_f32, -1.0, -1.0];
        let aabb_max = [1.0_f32, 1.0, 1.0];
        let atom = [1.0_f32, 0.0, 0.0]; // single atom on +x axis
        let alpha = 1.0_f32;

        // Expected magnitude per kl: |F| = α · Δ_AB · |x_i − X_c|
        //                               = 1   · Δ_AB · 1
        //                               = Δ_AB.
        let kl_values: [f32; 5] = [0.1, 1.0, 5.0, 10.0, 100.0];
        let mut prev_kl: Option<f32> = None;
        let mut prev_mag: Option<f32> = None;
        for &kl in kl_values.iter() {
            let forces = cpu_asc_reference(aabb_min, aabb_max, &[atom], &[true], kl, alpha, 1);
            let mag = forces[0].magnitude();
            assert!(
                (mag - kl).abs() < 1e-5,
                "expected |F| = Δ_AB ({}) for unit-distance atom, got {}",
                kl,
                mag,
            );
            if let (Some(p_kl), Some(p_mag)) = (prev_kl, prev_mag) {
                let kl_ratio = kl / p_kl;
                let mag_ratio = mag / p_mag;
                assert!(
                    (kl_ratio - mag_ratio).abs() < 1e-4,
                    "non-linear scaling: kl_ratio={}, mag_ratio={}",
                    kl_ratio,
                    mag_ratio,
                );
            }
            prev_kl = Some(kl);
            prev_mag = Some(mag);
        }
    }

    /// **Section K.3 — Prune / Violation codes produce zero force.**
    /// Defense in depth: even if the ASC kernel runs by mistake on a
    /// non-Construct route, no atomicAdds happen.
    #[test]
    fn asc_section_k3_inactive_for_prune_and_violation() {
        let aabb_min = [-1.0_f32, -1.0, -1.0];
        let aabb_max = [1.0_f32, 1.0, 1.0];
        let atom = [1.0_f32, 0.0, 0.0];
        for &code in &[0_u32, 2_u32] {
            let forces = cpu_asc_reference(aabb_min, aabb_max, &[atom], &[true], 1.0, 1.0, code);
            assert_eq!(forces[0].fx, 0.0, "code {} fx", code);
            assert_eq!(forces[0].fy, 0.0, "code {} fy", code);
            assert_eq!(forces[0].fz, 0.0, "code {} fz", code);
        }
    }

    /// **Section K.4 — cluster mask exclusion.** Atoms not in the
    /// cluster receive zero force (branchless mask multiply, no
    /// warp divergence on the GPU side).
    #[test]
    fn asc_section_k4_mask_excludes_non_cluster_atoms() {
        let aabb_min = [-1.0_f32, -1.0, -1.0];
        let aabb_max = [1.0_f32, 1.0, 1.0];
        let atoms: Vec<[f32; 3]> = vec![
            [1.0, 0.0, 0.0], // in cluster
            [0.0, 1.0, 0.0], // NOT in cluster
        ];
        let mask = [true, false];
        let forces = cpu_asc_reference(aabb_min, aabb_max, &atoms, &mask, 1.0, 1.0, 1);
        assert!(forces[0].fx > 0.0, "in-cluster atom must get nonzero force");
        assert_eq!(forces[1].fx, 0.0);
        assert_eq!(forces[1].fy, 0.0);
        assert_eq!(forces[1].fz, 0.0);
    }

    // ─── T4 — clock64 conversion + pipeline elapsed ─────────────────────

    #[test]
    fn cycles_to_ns_at_blackwell_boost() {
        // 2977 cycles at 2.977 GHz ≈ 1000 ns. Tolerance: 1 ns.
        let ns = cycles_to_ns(2977);
        assert!(
            (ns - 1000.0).abs() < 1.0,
            "2977 cycles → {} ns (expected ≈1000)",
            ns
        );

        // Round-trip via pipeline_elapsed.
        let mut adj = InterferometricAdjudicatorFfi::zero();
        adj.start_clock = 1_000;
        adj.stop_clock = 30_770; // 29770-cycle window
        let (cycles, ns) = pipeline_elapsed(&adj);
        assert_eq!(cycles, 29_770);
        // 29770 / 2.977 ≈ 9_999.66 ns — just under the 10 μs gate.
        assert!(
            (ns - 9_999.66).abs() < 1.0,
            "29770 cycles → {} ns (expected ≈9999.66)",
            ns
        );
    }

    #[test]
    fn pipeline_elapsed_saturates_on_clock_wraparound() {
        // If stop_clock < start_clock (clock wrap or out-of-order
        // bookend launch), saturating_sub returns 0 — never panics.
        let mut adj = InterferometricAdjudicatorFfi::zero();
        adj.start_clock = 10_000;
        adj.stop_clock = 5_000; // out of order
        let (cycles, ns) = pipeline_elapsed(&adj);
        assert_eq!(cycles, 0);
        assert_eq!(ns, 0.0);
    }

    #[test]
    fn pointer_stability_contract_documents_lifecycle() {
        // Sanity: the contract string is non-empty and references the
        // create / destroy / capture-region invariant.
        assert!(POINTER_STABILITY_CONTRACT.contains("OUTSIDE"));
        assert!(POINTER_STABILITY_CONTRACT.contains("create"));
        assert!(POINTER_STABILITY_CONTRACT.contains("malloc"));
    }

    // ─── T7 — Locked calibration regression guard ───────────────────────

    #[test]
    fn t7_constants_match_operator_published_values() {
        // Byte-exact regression guard: any in-place edit to
        // T7_CALIBRATED_MU / SIGMA fires here. Re-calibration must
        // introduce NEW const names rather than mutating these.
        // Operator-published values, locked 2026-04-30.
        assert_eq!(
            T7_CALIBRATED_MU,
            [
                0.8052561253_f32,
                0.0040383553_f32,
                0.0703344136_f32,
                0.0538048399_f32,
                0.0396647932_f32,
                0.0269014686_f32,
            ]
        );
        assert_eq!(
            T7_CALIBRATED_SIGMA,
            [
                0.1482125481_f32,
                0.0090773341_f32,
                0.0805278777_f32,
                0.0222988033_f32,
                0.0565869628_f32,
                0.0099504697_f32,
            ]
        );
    }

    #[test]
    fn t7_constants_yield_finite_3_sigma_thresholds() {
        // Sanity: μ + 3σ must be finite + positive for every band.
        // A degenerate σ ≤ 0 or NaN would silently degrade the
        // Adjudicator into "always-Construct" or "always-Prune".
        for l in 0..6 {
            let mu = T7_CALIBRATED_MU[l];
            let sigma = T7_CALIBRATED_SIGMA[l];
            assert!(mu.is_finite(), "μ_l={} non-finite at band {}", mu, l);
            assert!(sigma.is_finite(), "σ_l={} non-finite at band {}", sigma, l);
            assert!(
                sigma > 0.0,
                "σ_l={} non-positive at band {} (would degenerate threshold)",
                sigma,
                l
            );
            let threshold = mu + 3.0 * sigma;
            assert!(threshold.is_finite(), "threshold non-finite at band {}", l);
            assert!(threshold > 0.0, "threshold non-positive at band {}", l);
        }
    }

    #[test]
    fn t7_constants_round_trip_through_zero_struct() {
        // Round-trip: writing the calibrated values into a freshly
        // zero'd InterferometricAdjudicatorFfi struct via field
        // assignment yields a struct whose noise_floor_mu and
        // noise_floor_sigma fields contain the calibrated values
        // bit-exactly. This validates that the FFI struct's f32
        // fields can hold the operator-published precision without
        // precision loss (every value is representable as f32).
        let mut adj = InterferometricAdjudicatorFfi::zero();
        adj.noise_floor_mu.copy_from_slice(&T7_CALIBRATED_MU);
        adj.noise_floor_sigma.copy_from_slice(&T7_CALIBRATED_SIGMA);
        assert_eq!(adj.noise_floor_mu, T7_CALIBRATED_MU);
        assert_eq!(adj.noise_floor_sigma, T7_CALIBRATED_SIGMA);

        // Cross-check the field offsets the FFI memcpy will hit.
        assert_eq!(offset_of!(InterferometricAdjudicatorFfi, noise_floor_mu), 0);
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, noise_floor_sigma),
            24
        );
    }

    #[test]
    fn t7_constants_band_0_dominates_as_expected() {
        // Sanity on physical reasonableness: l=0 (the SO(3) total
        // power) should dominate the higher bands, matching the
        // expected Σ KL energy distribution where most of the
        // information lives in the rotationally-invariant scalar
        // mode. l=0 μ ≈ 0.805, all other bands < 0.1 ⇒ ≥ 8× ratio.
        let mu_l0 = T7_CALIBRATED_MU[0];
        for l in 1..6 {
            let mu_l = T7_CALIBRATED_MU[l];
            assert!(
                mu_l0 > mu_l * 8.0,
                "expected l=0 μ to dominate l={} by ≥ 8×; got μ_0={} μ_{}={}",
                l,
                mu_l0,
                l,
                mu_l,
            );
        }
    }

    // ─── G19 — F1 SWITCH predicate sub-byte address invariants ──────────

    #[test]
    fn g19_adjudication_code_offset_is_pinned_at_52() {
        // The F1 SWITCH predicate address = struct_base + 52. This
        // offset is the FFI contract; drift here would mean the
        // hardware scheduler reads the wrong word.
        assert_eq!(
            offset_of!(InterferometricAdjudicatorFfi, adjudication_code),
            ADJUDICATION_CODE_OFFSET,
            "G19 invariant: adjudication_code MUST remain at byte offset 52"
        );
        assert_eq!(ADJUDICATION_CODE_OFFSET, 52);
    }

    #[test]
    fn g19_predicate_address_4_byte_aligned() {
        // 52 % 4 == 0 ⇒ the field address is 4-aligned regardless of
        // the struct's 128-byte base alignment. Required for atomic
        // LDG.E.32 read by the hardware scheduler.
        assert_eq!(ADJUDICATION_CODE_OFFSET % 4, 0);
    }

    #[test]
    fn g19_predicate_within_one_l1_sector() {
        // The struct is 128 bytes, fits entirely in one Blackwell L1
        // sector starting at the struct's 128-aligned base address.
        // The field at offset 52 + its 4 bytes (ending at 56) is well
        // within the 0..128 sector range. NO split-load penalty.
        let field_end = ADJUDICATION_CODE_OFFSET + std::mem::size_of::<u32>();
        assert!(field_end <= 128,
            "adjudication_code (offset {}, size 4) MUST NOT cross the 128-byte sector boundary (would force split-load)",
            ADJUDICATION_CODE_OFFSET);
    }

    #[test]
    fn g19_devptr_returns_field_address() {
        // adjudication_code_devptr(&adj) must return adj_base + 52.
        let mut adj = InterferometricAdjudicatorFfi::zero();
        let base = &adj as *const _ as usize;
        let field_addr = unsafe { adjudication_code_devptr(&adj as *const _) as usize };
        assert_eq!(
            field_addr - base,
            ADJUDICATION_CODE_OFFSET,
            "devptr offset drift: got {} expected {}",
            field_addr - base,
            ADJUDICATION_CODE_OFFSET,
        );
        // The devptr lets the caller actually read the value.
        adj.adjudication_code = 1; // Construct
        let read_back = unsafe { *adjudication_code_devptr(&adj as *const _) };
        assert_eq!(read_back, 1);
    }

    #[test]
    fn g19_predicate_address_stable_across_repeated_access() {
        // Pointer stability: repeated calls to adjudication_code_devptr
        // on the same struct return the same address. (Trivially true
        // for a fixed allocation, but pinned here as the contract.)
        let adj = InterferometricAdjudicatorFfi::zero();
        let p1 = unsafe { adjudication_code_devptr(&adj as *const _) };
        let p2 = unsafe { adjudication_code_devptr(&adj as *const _) };
        assert_eq!(p1, p2, "predicate address drifted between calls");
    }
}
