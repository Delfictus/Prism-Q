//! M1.2.19.B / Amendment 3.13 — Asynchronous Manifold Sequencer **Channel B**
//! (GhostTileFrame stream).
//!
//! The previous V2 telemetry pipeline persisted only **end-of-run snapshots**
//! (phasor_kcc_state.json, spatial_grid_state.json, etc.) — adequate for
//! steady-state diagnostics but useless for the per-frame causal/coherence
//! analysis the operator's Bilateral Phase-Coherence (Φ_sym) and
//! Flux-Coupling Efficiency (η) metrics require.  Channel B closes that gap:
//! on each captured-graph replay the GPU pushes a self-describing 4096-byte
//! record for every cluster that passes the v2 emission gate —
//! `adjudication_code >= 1` (Construct or Violation), OR
//! `kl_divergence >= observation_threshold_kl` (observation pass), OR the
//! diagnostic firehose with physically-active KL
//! (`cuda/ghost_tile_kernel.cu:534-538`) — into a
//! pinned-host, device-mapped staging ring. Records emitted by the
//! observation or firehose gate carry their true `adjudication_code`,
//! including 0. At synchronized replay boundaries
//! the host durably commits hash-chained segments and only then resets the
//! ring. The ring and legacy flat file are not artifact authorities.
//!
//! ## Record layout (per active cluster per frame)
//!
//! v9D' (Amendment 3.14 / Sector-Lock): each record is a single 4096-byte
//! GhostTileFrame. The size is one NVMe physical sector, so record offsets
//! are exact sector multiples and `GHOST_RECORD_BYTES` arithmetic is
//! alignment-free. Durability is buffered write + `File::sync_all` + rename
//! + directory fsync in `GhostSegmentWriter::append_payload`
//! (`ghost_segment.rs:779-796`) — NOT `O_DIRECT`. (The ZSTR consumer does
//! open with `O_DIRECT | O_DSYNC`, `zstr.rs:862`; this path does not.)
//! The v3 producer uses the
//! former alignment region at offsets 256..4096 for exact RichSpike
//! cluster-residue membership counts. These counts describe the spike
//! population assigned to the cluster; they are not cavity-lining contacts.
//!
//! ```text
//! offset  size  field
//! ──────  ────  ──────
//!     0  4096   GhostTileFrame header  (this struct, sector-aligned)
//! ──────  ────
//!  total 4096   bytes per record  (= one NVMe physical sector)
//! ```
//!
//! ## GhostTileFrame header (4096 bytes, align(4096))
//!
//! Operator spec (Amendment 3.14 §1.1 — v9D' Sector-Lock):
//!
//! ```text
//! offset  size  field                  purpose
//! ──────  ────  ────────────────────   ──────────────────────────────────
//!     0     8   frame_idx u64          capture-identity tag (graph-build constant)
//!     8     4   site_id   u32          cluster index within frame
//!    12     1   chain_id  u8           numeric role 0=A, 1=B; historical ASCII A/B accepted by scanner
//!    13     1   adjudication_code u8   0=Prune, 1=Construct, 2=Violation
//!    14     2   telemetry_flags u16    bit-packed: LQI, T7, Burst, Drift
//!    16     4   kl_divergence f32      Σ_planes Δ_AB at this frame
//!    20    96   power_spectrum [f32;24]  4 planes × 6 bands (L=0..5)
//!   116     8   thermo_flux    [f32;2] [therm C_l=0, caus C_l=0]
//!   124     4   causal_lead_residue u32  driver_id for Lag Persistence
//!   128   128   _reserved_payload [u32;32]  v2 MAR overlay (schema, passes,
//!                                            gear, dt, step, AABB, centroid)
//!   256    64   v3 membership header    counts, flags, coordinate-sidecar slot
//!   320  3776   v3 membership entries   472 × (residue_id, spike_count)
//! ```
//!
//! ## Why mapped pinned memory (operator §1.2 Mapped Memory Standard)
//!
//! The buffer is allocated via `cuMemHostAlloc(.., PORTABLE | DEVICEMAP)`,
//! which produces page-locked host memory addressable from both host and
//! GPU through a single virtual-address window.  The captured kernel
//! writes through the device-mapped pointer; the host reads through the
//! host pointer with no `cudaMemcpy` round-trip.  On Blackwell sm_120 +
//! PCIe Gen5 this lands as direct DMA from L2 to host memory.

#![cfg(any(feature = "gpu", feature = "cpu-analysis"))]

#[cfg(feature = "gpu")]
use cudarc::driver::sys::{
    cuMemFreeHost, cuMemHostAlloc, cuMemHostGetDevicePointer_v2, CUresult,
    CU_MEMHOSTALLOC_DEVICEMAP, CU_MEMHOSTALLOC_PORTABLE,
};
use sha2::{Digest, Sha256};
#[cfg(feature = "gpu")]
use std::ffi::c_void;

// ─── GhostTileFrame — 4096-byte aligned header (v9D' Sector-Lock) ────────────

/// Self-describing temporal/site tag for the Channel-B exfiltration
/// stream.  v9D' (Amendment 3.14): the frame is sized to exactly one NVMe
/// physical sector (4096 bytes, align 4096) so every record offset is an
/// exact NVMe sector multiple and `GHOST_RECORD_BYTES` arithmetic is
/// alignment-free.  (This path does not use `O_DIRECT` — see the module
/// doc.)
///
/// Layout pinned by `static_assert`s on the C++ mirror in
/// `cuda/ghost_tile_kernel.cuh` and the const-context offset asserts at
/// the bottom of this module.  Any drift breaks the offline replay.
#[repr(C, align(4096))]
#[derive(Debug, Clone, Copy)]
pub struct GhostTileFrame {
    /// Capture-identity tag, NOT a monotonic clock.  Set from the
    /// kernel-node scalar argument `initial_frame_id`, baked at
    /// captured-graph build time (`captured_pipeline.rs:4602-4604`, passed
    /// at `:4840` and `:4863`).  Every replay of one captured graph writes
    /// the SAME value; it changes only when the host re-records the graph.
    /// Nothing in either kernel increments it — both write it verbatim
    /// (`cuda/ghost_tile_kernel.cu:154` and `:554`).
    ///
    /// The authoritative time axis is the v2 `step_idx` field at offset 152,
    /// fed by the device-side timekeeper
    /// (`cuda/ghost_tile_kernel.cu:628-637`).  Φ_sym and Lag-Persistence
    /// integrals MUST use `step_idx`; integrating against `frame_idx`
    /// integrates against a constant and yields a smooth, finite, entirely
    /// fabricated time axis.  `ghost_phase_materializer.rs` already orders by
    /// `step_idx` first.
    pub frame_idx: u64,

    /// Cluster index within the frame (0..n_clusters).  For 7C8R dimer
    /// with n_clusters=1, always 0.  Multi-cluster targets (KRAS, MYC)
    /// will produce one record per active cluster.
    pub site_id: u32,

    /// Compact chain role emitted by CUDA: 0=A, 1=B. Historical artifacts may
    /// use ASCII 0x41/0x42; the evidence scanner accepts both encodings.
    /// Resolved on device: the Path-Ω geometry-emergent LUT
    /// `d_cluster_to_chain_id[i]` when populated, else a residue-boundary
    /// scan of this cluster's representative residue against
    /// `d_chain_offsets` (`cuda/ghost_tile_kernel.cu:175-188` for v1 and
    /// `:556-569` for v2 — the only two reads).  See
    /// `set_cluster_repr_residue` below for the LUT contract and for what
    /// each unpopulated LUT costs.
    pub chain_id: u8,

    /// Mirror of the F1 SWITCH selector: 0=Prune, 1=Construct, 2=Violation.
    /// NOT filtered to >= 1: the v2 observation gate and the diagnostic
    /// firehose both emit records carrying code 0
    /// (`cuda/ghost_tile_kernel.cu:534-538`, `cuda/ghost_tile_kernel.cuh:210-215`).
    /// Consumers counting Construct events MUST filter on this field.
    /// Stored so the offline replay can distinguish Construct
    /// (legitimate burst) from Violation (numerical trap) from a
    /// diagnostic/observation sample.
    pub adjudication_code: u8,

    /// **Wave 1 / Q1+P4** — bitfield of per-record telemetry flags.
    /// Bit 0 (`CLASS_TAINTED` = 0x0001): set when the kernel substituted 0.0
    ///   for a non-finite input anywhere in this record's measured payload.
    ///   Complete contributor list (`cuda/ghost_tile_kernel.cu:215`, `:224`
    ///   and `:591`, `:596`):
    ///     - any of the 24 `power_spectrum` bands (4 planes × 6) non-finite;
    ///     - `thermo_flux[0]` source `therm_power_spectrum[0]` non-finite;
    ///     - `thermo_flux[1]` source `caus_power_spectrum[0]` non-finite.
    ///   A finite `thermo_flux` pair does NOT imply an untainted record.
    /// Bits 1..15 reserved for future quality flags (chain-resolution
    ///   confidence, KCC-stale, gear-id snapshot, etc.).
    /// Replaces the alignment-only `_pad: [u8; 2]` placeholder; offset
    /// and size unchanged so the on-disk layout is bit-for-bit
    /// equivalent for records produced before the flag was wired.
    pub telemetry_flags: u16,

    /// Σ_planes Δ_AB — total weighted KL divergence across all 4 planes
    /// at this frame.  Mirrors `adj.current_divergence` at the moment
    /// the adjudicator wrote the SWITCH code.
    pub kl_divergence: f32,

    /// 4-plane SO(3) power spectrum: `C_l[0..6]` for each of
    /// {geometry, causal, thermo, chemistry} = 24 floats total.
    /// Layout: `power_spectrum[plane * 6 + l]` for `plane ∈ 0..4`,
    /// `l ∈ 0..6`.  Φ_sym integrates plane 0 against the Chain-B
    /// partner's spectrum across time; planes 1..3 feed the
    /// 4-plane interferometric KL.  All four planes are wired
    /// (M1.2.25 plane-fan-out): `cuda/ghost_tile_kernel.cu:206-216`
    /// and `:582-592` copy geo/caus/therm/chem out of the
    /// ContactShellTile into stripes 0/6/12/18.
    ///
    /// ENCODING — a structurally-empty band is **not** 0.0, and not
    /// NaN.  The SO(3) producer epsilon-pads every C_l:
    /// `cuda/so3_project.cu:503` declares `KL_EPS = 1e-7f` and `:525`
    /// applies `fmaxf(c_l, KL_EPS)`, and the empty-cluster fast path
    /// at `cuda/so3_project.cu:213-219` fills all four plane spectra
    /// with the same 1e-7 pad.  The ghost kernel copies those values
    /// through unchanged; it writes a literal 0.0 only when the input
    /// is non-finite, and then also sets `CLASS_TAINTED`.  So a plane
    /// carrying no power reads as 1e-7 in every band.
    ///
    /// Consumers MUST NOT infer emptiness by comparing a band against
    /// 0.0 — `!= 0.0` classifies the 1e-7 pad as populated.  Three
    /// decoders do exactly that today (`ghost_phase_lattice.rs:781`,
    /// `bin/prism_ghost_evidence_scan.rs:764` and `:1983`); the
    /// encoding repair is a separate phase and is NOT applied here.
    pub power_spectrum: [f32; 24],

    /// `[therm_C0, caus_C0]` — the l=0 band of the thermo plane and the l=0
    /// band of the causal plane, copied from `ContactShellTile`
    /// (`cuda/ghost_tile_kernel.cu:222-226` for v1 and `:594-598` for v2).
    /// The kernel writes `0.0f`, never NaN, when a source is non-finite, and
    /// sets `CLASS_TAINTED`.
    ///
    /// NOT a water-density derivative and NOT a vibrational-energy snapshot:
    /// both slots are SO(3) power-spectrum bands, and the kernel itself
    /// labels the second one "causal trigger l=0" (`:219`).  The
    /// Flux-Coupling Efficiency η = ΔV_wd / ΔV_vib requires a
    /// desolvation/vibrational bus that is not wired to this field; do not
    /// compute η from these two slots.
    pub thermo_flux: [f32; 2],

    /// Driver residue id from the KCC causal plane — the per-cluster
    /// `argmax|d_kcc_temporal_corr|` computed host-side between chunks and
    /// staged in the F2-pool `d_kcc_lead` buffer
    /// (`captured_pipeline.rs:1556-1570`, selected at `:4620` and passed to
    /// the launchers at `:4839` and `:4862`).  The kernel copies
    /// `d_kcc_lead[i]` when that buffer is present and writes the
    /// `0xFFFFFFFF` "not resolved at GPU time" sentinel when it is absent —
    /// the normal bootstrap state before the first KCC chunk completes
    /// (`cuda/ghost_tile_kernel.cu:231-233` and `:600`).  The sentinel is the
    /// correct value; see `set_cluster_repr_residue` for why it must never be
    /// back-filled from the representative-residue LUT.
    pub causal_lead_residue: u32,

    /// Legacy field name for the 128-byte span at offset 128.  In v2/v3 this
    /// region is the structured MAR overlay — `schema_version`,
    /// observation / discovery pass, perturbation channel, uv wavelength,
    /// `field_completeness_flags`, `gear_id`, `dt_fs`, `step_idx`, AABB,
    /// centroid — enumerated in the "M1.2.23 §5" offset table below.  It is
    /// NOT reserved and `gear_id` is NOT a future field: it is live at offset
    /// 140 with a full sentinel discipline (`cuda/ghost_tile_kernel.cu:626`,
    /// `:618-625`; `cuda/ghost_tile_kernel.cuh:142-150`).  The v2 kernel
    /// zeroes all 32 words and then writes the overlay
    /// (`cuda/ghost_tile_kernel.cu:605-637`, `:663-688`).  The legacy v1
    /// kernel zeroes the span and writes nothing, so `schema_version`
    /// reads 0.
    ///
    /// Allocation is NOT an initialization guarantee: `cuMemHostAlloc` does
    /// not promise zero-filled pinned memory.  `zero_ghost_ring_allocation`
    /// is the only initialization authority and must run before the device
    /// alias is published.
    pub _reserved_payload: [u32; 32],

    /// Legacy field name retained for ABI stability. In v3 these bytes carry
    /// the fixed cluster-residue membership header and bounded exact entries;
    /// consumers must decode them through the v3 offsets below. The complete
    /// struct remains one 4096-byte sector.
    pub _slack: [u8; 3840],
}

/// **Wave 1 / P4** — telemetry_flags bit definitions.
///
/// Bit 0 — `CLASS_TAINTED`: set when the Ghost stage kernel substituted 0.0
/// for a non-finite input anywhere in this record's measured payload.
/// Complete contributor list (`cuda/ghost_tile_kernel.cu:215`, `:224` for the
/// v1 kernel and `:591`, `:596` for the v2 kernel):
///   - any of the 24 `power_spectrum` bands (4 planes × 6) non-finite;
///   - `thermo_flux[0]`, source `therm_power_spectrum[0]`, non-finite;
///   - `thermo_flux[1]`, source `caus_power_spectrum[0]`, non-finite.
///
/// It is a record-wide numerical-hygiene flag, not a `thermo_flux`-only flag,
/// and it does not indicate an unwired upstream slot: both `thermo_flux`
/// slots are written on every record. A finite `thermo_flux` pair therefore
/// does NOT imply an untainted record. Offline integrators MUST exclude
/// tainted records from any spectral or flux quantity.
pub const GHOST_TELEMETRY_CLASS_TAINTED: u16 = 0x0001;

impl GhostTileFrame {
    /// Zero-valued seed with an explicit unavailable causal-lead sentinel.
    /// It is NOT a model
    /// of allocator state: `cuMemHostAlloc` does not promise zeroed pinned
    /// memory — see `zero_ghost_ring_allocation` below, which exists solely
    /// because of that.  Nor do kernels write these fields atomically: the
    /// only atomic is the ring-slot claim (`cuda/ghost_tile_kernel.cu:138`
    /// for v1, `:542` for v2), after which the 4096-byte record is written
    /// with ordinary stores.  Record visibility to the host is established by
    /// the caller's stream synchronization, not by the write itself — see
    /// `persist_and_reset_after_stream_sync`.
    pub const fn zero() -> Self {
        Self {
            frame_idx: 0,
            site_id: 0,
            chain_id: 0,
            adjudication_code: 0,
            telemetry_flags: 0,
            kl_divergence: 0.0,
            power_spectrum: [0.0; 24],
            thermo_flux: [0.0; 2],
            causal_lead_residue: u32::MAX,
            _reserved_payload: [0u32; 32],
            _slack: [0u8; 3840],
        }
    }
}

unsafe impl Send for GhostTileFrame {}
unsafe impl Sync for GhostTileFrame {}

// Compile-time layout invariants (operator spec Amendment 3.14 §1.1 — v9D').
const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<GhostTileFrame>() == 4096);
    assert!(align_of::<GhostTileFrame>() == 4096);
    assert!(offset_of!(GhostTileFrame, frame_idx) == 0);
    assert!(offset_of!(GhostTileFrame, site_id) == 8);
    assert!(offset_of!(GhostTileFrame, chain_id) == 12);
    assert!(offset_of!(GhostTileFrame, adjudication_code) == 13);
    assert!(offset_of!(GhostTileFrame, telemetry_flags) == 14);
    assert!(offset_of!(GhostTileFrame, kl_divergence) == 16);
    assert!(offset_of!(GhostTileFrame, power_spectrum) == 20);
    assert!(offset_of!(GhostTileFrame, thermo_flux) == 116);
    assert!(offset_of!(GhostTileFrame, causal_lead_residue) == 124);
    assert!(offset_of!(GhostTileFrame, _reserved_payload) == 128);
    assert!(offset_of!(GhostTileFrame, _slack) == 256);
};

/// Per-record byte size: 4096 (one sector, header-only — ContactShellTile
/// retired from on-disk format in v9D' / Amendment 3.14).
pub const GHOST_RECORD_BYTES: usize = 4096;

// ─── M1.2.23 §5 — GhostTileFrame v2 MAR payload schema overlay ──────────────
//
// The 128-byte `_reserved_payload [u32; 32]` region at offset 128 is partitioned
// into structured Transparent-MAR fields. Backward compatibility is preserved:
//
//   * v1 producer zeros the whole 128-byte region (legacy kernel behavior)
//     → schema_version reads as 0 → consumers treat as legacy v1.
//
//   * The production v2/v3 kernel writes GHOST_FRAME_SCHEMA_V3 = 3 and
//     populates both the overlay at 128..256 and the v3 membership region
//     at 256..4096 (cuda/ghost_tile_kernel.cu:609).
//
// GHOST_FRAME_SCHEMA_V2 = 2 is DEFINED BUT UNWRITTEN: no producer in this
// tree emits it. It is retained so a "2" reading is recognizable rather than
// unclassifiable. Consumers gating on the production stream MUST match V3.
//
// On-disk layout is BYTE-IDENTICAL between v1 and v2; the only producer-side
// change is what bytes get written into the reserved region. No struct-field
// rename. No size change. No alignment change. CUDA mirror remains a 4096-byte
// align(4096) struct with the same field list — the v2 fields are accessed
// via byte offsets / typed views below.
//
// Schema versioning policy:
//   * GHOST_FRAME_SCHEMA_V1_LEGACY = 0 (zeros — no v2 fields populated)
//   * GHOST_FRAME_SCHEMA_V2        = 2 (defined, never written by any producer)
//   * GHOST_FRAME_SCHEMA_V3        = 3 (the value the production kernel writes)
// schema_version=1 is RESERVED (never used) so a "1" reading is unambiguous
// drift / corruption and the scanner can flag it.
//
// Field layout within the 128-byte reserved region (offsets relative to
// the START of the GhostTileFrame, not relative to the reserved region):
//
//   offset 128  schema_version          u32   (0=v1 legacy, 2=v2 explicit)
//   offset 132  observation_pass        u8    (0/1 — bool)
//   offset 133  discovery_pass          u8    (0/1 — bool)
//   offset 134  perturbation_channel    u8    (UV bit code 0..3, or 0xFF unknown)
//   offset 135  _pad8                   u8
//   offset 136  uv_wavelength_nm        u16   (260, 280, 305, 320, or 0)
//   offset 138  field_completeness_flags u16
//   offset 140  gear_id                 u32   (Wave A default 0)
//   offset 144  dt_fs                   f32
//   offset 148  _pad32_for_step_align   u32   (slot reserved so step_idx u64 lands 8-aligned)
//   offset 152  step_idx                u64   (8-aligned — required for STG.E.64 stores)
//   offset 160  aabb_min                [f32; 3]   (160, 164, 168)
//   offset 172  aabb_max                [f32; 3]   (172, 176, 180)
//   offset 184  centroid_xyz            [f32; 3]   (184, 188, 192)
//   offset 196  _v2_reserved            [u32; 15]  (196..256, 60 B for future)
//
//   offset 256  v3 membership region [u8; 3840] (legacy field name `_slack`)
//
// M1.2.24 fix: step_idx was at 148 in the v2 schema landed by Commit 3.
// 148 mod 8 = 4 → not 8-aligned → CUDA STG.E.64 traps with
// CUDA_ERROR_MISALIGNED_ADDRESS (deferred). The misaligned u64 store in
// the v2 kernel was the root cause of the M1.2.24 segfault. Swapping the
// pad and step_idx so step_idx lands at 152 (152 mod 8 = 0) fixes the
// alignment. No on-disk format compat issue exists because v2 records
// were never written (smoke aborted before any kernel completed).
//
// The v2 payload uses offsets 128..256. v3 additionally interprets the legacy
// `_slack` field at 256..4096 as cluster-residue membership. CUDA mirror's
// `_reserved_payload[32]` still spans the same first 128 bytes.

pub const GHOST_FRAME_SCHEMA_V1_LEGACY: u32 = 0;
pub const GHOST_FRAME_SCHEMA_V2: u32 = 2;
/// Production cluster-residue membership schema. Preserves the 4096-byte
/// record while using the former slack region for exact RichSpike membership.
pub const GHOST_FRAME_SCHEMA_V3: u32 = 3;

pub const GHOST_V2_OFFSET_SCHEMA_VERSION: usize = 128;
pub const GHOST_V2_OFFSET_OBSERVATION_PASS: usize = 132;
pub const GHOST_V2_OFFSET_DISCOVERY_PASS: usize = 133;
pub const GHOST_V2_OFFSET_PERTURBATION_CHAN: usize = 134;
pub const GHOST_V2_OFFSET_UV_WAVELENGTH_NM: usize = 136;
pub const GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS: usize = 138;
pub const GHOST_V2_OFFSET_GEAR_ID: usize = 140;
pub const GHOST_V2_OFFSET_DT_FS: usize = 144;
pub const GHOST_V2_OFFSET_STEP_IDX: usize = 152;
pub const GHOST_V2_OFFSET_AABB_MIN: usize = 160;
pub const GHOST_V2_OFFSET_AABB_MAX: usize = 172;
pub const GHOST_V2_OFFSET_CENTROID: usize = 184;
pub const GHOST_V2_OFFSET_V2_RESERVED: usize = 196;

/// P4(B) — bit set when the producer wrote a plane-populated mask at all.
/// Distinguishes "all four planes empty" from "this producer predates the mask".
pub const GHOST_PLANE_MASK_PRESENT: u32 = 1 << 8;
/// Plane bits within the mask: 0 Geometry, 1 Causality, 2 Thermodynamics, 3 Chemistry.
pub const GHOST_PLANE_BIT_GEOMETRY: u32 = 1 << 0;
pub const GHOST_PLANE_BIT_CAUSALITY: u32 = 1 << 1;
pub const GHOST_PLANE_BIT_THERMO: u32 = 1 << 2;
pub const GHOST_PLANE_BIT_CHEMISTRY: u32 = 1 << 3;

/// Decode the exact plane-populated mask from a ghost record.
///
/// Returns `None` for a record written before the mask existed, so callers can
/// fall back deliberately rather than silently reading zeros as "all empty".
///
/// Why this exists: the SO(3) spectra are padded with KL_EPS (1e-7) so that
/// log(0) never reaches the adjudicator, which hard-traps on non-finite values
/// in the Causality and Thermodynamics planes. That pad is correct and stays.
/// But it means a structurally empty plane does NOT read as 0.0, so any consumer
/// inferring emptiness by testing the spectrum against zero concludes the plane
/// is populated. The producer knows the truth exactly — a plane is empty iff its
/// `sum_w` is zero — so it now publishes that as a mask, and consumers read the
/// declared fact instead of re-deriving it from a deliberately non-zero value.
pub fn ghost_plane_populated_mask(record: &[u8]) -> Option<u32> {
    let off = GHOST_V2_OFFSET_V2_RESERVED;
    if record.len() < off + 4 {
        return None;
    }
    let raw = u32::from_le_bytes(record[off..off + 4].try_into().ok()?);
    if raw & GHOST_PLANE_MASK_PRESENT == 0 {
        return None;
    }
    Some(raw)
}

/// Explicit separation of producer occupancy from observed spectral content.
/// A legacy mask-less floor vector cannot establish producer occupancy and is
/// excluded from cosine evidence; values are never rewritten or zero-filled.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GhostPlaneEvidence {
    pub producer_populated_mask: Option<u32>,
    pub exact_floor_only_mask: u8,
    pub above_floor_observation_mask: u8,
    pub below_known_padding_floor_mask: u8,
    pub nonfinite_or_negative_mask: u8,
    pub cosine_evidence_mask: u8,
    pub interpretation: String,
}
pub fn ghost_plane_evidence(record: &[u8]) -> Option<GhostPlaneEvidence> {
    if record.len()<116 {return None;}
    let producer=ghost_plane_populated_mask(record);
    let mut floor=0;let mut above=0;let mut below=0;let mut invalid=0;
    for plane in 0..4 {
        let values:Vec<f32>=(0..6).map(|band|{let o=20+(plane*6+band)*4;f32::from_le_bytes(record[o..o+4].try_into().unwrap())}).collect();
        if values.iter().all(|v|v.to_bits()==1.0e-7f32.to_bits()){floor|=1<<plane;}
        if values.iter().any(|v|v.is_finite()&&*v<1.0e-7f32){below|=1<<plane;}
        if values.iter().any(|v|!v.is_finite()||*v<0.0){invalid|=1<<plane;}
        if values.iter().all(|v|v.is_finite())&&values.iter().any(|v|*v>1.0e-7f32){above|=1<<plane;}
    }
    let cosine=producer.map(|m|m as u8&0x0f).unwrap_or(above)&!invalid&!floor;
    Some(GhostPlaneEvidence{producer_populated_mask:producer,exact_floor_only_mask:floor,above_floor_observation_mask:above,below_known_padding_floor_mask:below,nonfinite_or_negative_mask:invalid,cosine_evidence_mask:cosine,interpretation:if producer.is_some(){"producer_mask_with_observed_floor_and_invalid_exclusion"}else{"producer_occupancy_unknown; above_floor_observations_only; floor_padding_is_not_coupling"}.into()})
}

// ─── v3 exact cluster-residue membership (former alignment region) ───────
// 256..320 is a fixed header; 320..4096 contains 472 × 8-byte entries.
pub const GHOST_V3_OFFSET_MEMBERSHIP_SCHEMA_VERSION: usize = 256;
pub const GHOST_V3_OFFSET_CLUSTER_RESIDUE_COUNT: usize = 260;
pub const GHOST_V3_OFFSET_CLUSTER_RESIDUE_CAPACITY: usize = 264;
pub const GHOST_V3_OFFSET_CLUSTER_RESIDUE_OVERFLOW: usize = 268;
pub const GHOST_V3_OFFSET_SOURCE_SPIKE_COUNT: usize = 272;
pub const GHOST_V3_OFFSET_UNRESOLVED_RESIDUE_SPIKE_COUNT: usize = 276;
pub const GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRY_BYTES: usize = 280;
pub const GHOST_V3_OFFSET_GEOMETRY_AVAILABILITY_FLAGS: usize = 284;
pub const GHOST_V3_OFFSET_COORDINATE_FRAME_SHA256: usize = 288;
pub const GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES: usize = 320;
pub const GHOST_V3_MEMBERSHIP_SCHEMA_VERSION: u32 = 1;
pub const GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES: usize = 8;
pub const GHOST_V3_CLUSTER_RESIDUE_CAPACITY: usize = (GHOST_RECORD_BYTES
    - GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES)
    / GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES;
pub const GHOST_V3_GEOMETRY_CLUSTER_RESIDUE_MEMBERSHIP_EXACT: u32 = 0x0001;
pub const GHOST_V3_GEOMETRY_COORDINATE_SIDECAR_REQUIRED: u32 = 0x0002;

/// SHA-256 identity of the exact record ABI and its scientific semantics.
/// This is embedded into every durable segment context so a loader cannot
/// silently join bytes produced under a different layout or interpretation.
pub fn ghost_record_abi_sha256() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"prism-ghost-record-abi-v3\0");
    hasher.update(b"rich_spike_cluster_membership_not_cavity_lining\0");
    for value in [
        GHOST_RECORD_BYTES as u64,
        GHOST_FRAME_SCHEMA_V3 as u64,
        GHOST_V2_OFFSET_SCHEMA_VERSION as u64,
        GHOST_V2_OFFSET_STEP_IDX as u64,
        GHOST_V2_OFFSET_AABB_MIN as u64,
        GHOST_V2_OFFSET_AABB_MAX as u64,
        GHOST_V2_OFFSET_CENTROID as u64,
        GHOST_V3_OFFSET_MEMBERSHIP_SCHEMA_VERSION as u64,
        GHOST_V3_OFFSET_CLUSTER_RESIDUE_COUNT as u64,
        GHOST_V3_OFFSET_CLUSTER_RESIDUE_CAPACITY as u64,
        GHOST_V3_OFFSET_CLUSTER_RESIDUE_OVERFLOW as u64,
        GHOST_V3_OFFSET_SOURCE_SPIKE_COUNT as u64,
        GHOST_V3_OFFSET_UNRESOLVED_RESIDUE_SPIKE_COUNT as u64,
        GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRY_BYTES as u64,
        GHOST_V3_OFFSET_GEOMETRY_AVAILABILITY_FLAGS as u64,
        GHOST_V3_OFFSET_COORDINATE_FRAME_SHA256 as u64,
        GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES as u64,
        GHOST_V3_MEMBERSHIP_SCHEMA_VERSION as u64,
        GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES as u64,
        GHOST_V3_CLUSTER_RESIDUE_CAPACITY as u64,
    ] {
        hasher.update(value.to_le_bytes());
    }
    ghost_sha256_hex(&hasher.finalize())
}

/// SHA-256 identity for the documented MAR-v1 segment-only record contract.
/// It binds only the legacy base frame and explicitly excludes an exact device
/// step, AABB/centroid projection, RichSpike membership, coordinate sidecar,
/// and Ghost-lattice eligibility.
pub fn ghost_legacy_record_abi_sha256() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"prism-ghost-record-abi-v1-legacy-segment-only\0");
    hasher.update(b"base_frame_no_exact_step_no_spatial_projection_no_membership_no_coordinate\0");
    for value in [
        GHOST_RECORD_BYTES as u64,
        GHOST_FRAME_SCHEMA_V1_LEGACY as u64,
        0,   // frame_idx
        8,   // site_id
        12,  // chain_id
        13,  // adjudication_code
        14,  // telemetry_flags
        16,  // kl_divergence
        20,  // power_spectrum
        116, // thermo_flux
        124, // causal_lead_residue
        GHOST_V2_OFFSET_SCHEMA_VERSION as u64,
    ] {
        hasher.update(value.to_le_bytes());
    }
    ghost_sha256_hex(&hasher.finalize())
}

/// SHA-256 over the four sources that define the Ghost record ABI and its
/// on-device producer: this module, the CUDA mirror header, the emission
/// kernel, and the captured-graph launcher. `include_bytes!` makes the exact
/// source bytes part of the binary, so the context stays portable when the
/// source tree is absent.
///
/// SCOPE BOUNDARY — this digest covers the RECORD SURFACE, not the full
/// producer/consumer chain. It deliberately excludes `cuda/so3_project.cu`
/// and `.cuh` (which define `ContactShellTile` and compute every
/// `power_spectrum` band, both `thermo_flux` slots and the AABB),
/// `ghost_segment.rs` (the durable writer), and every offline decoder. A
/// numerical change in `so3_project.cu` — `KL_EPS` at `:503`, the
/// empty-cluster pad at `:213-219`, the L2 normalization — changes what these
/// fields MEAN and does NOT move this digest; conversely any byte change in
/// the four member files, comments included, mints a new producer identity.
/// Widening the member set is a versioned migration, not an edit.
///
/// Runtime consumers compare this value against an artifact's recorded
/// `record_source_sha256` at `bin/nhs_rt_full.rs:506` and `:597` and
/// `coupled_md.rs:18596`. All three are run-scoped self-identity assertions —
/// the same `||` disjunction also requires `selected_engine_sha256` to equal
/// the running executable's own hash (`bin/nhs_rt_full.rs:504`, `:595`) or
/// `run_id` to equal the current run (`coupled_md.rs:18585`) — so this digest
/// is NOT an archive-compatibility gate, and no archived segment's
/// admissibility depends on it. Producer-epoch transitions are recorded in
/// `OPERATOR_CONTROL/GHOST_RECORD_SOURCE_EPOCH_20260921T003933Z/`.
pub fn ghost_record_source_sha256() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"prism-ghost-record-source-v1\0");
    for (label, source) in [
        (
            b"ghost_tile.rs".as_slice(),
            include_bytes!("ghost_tile.rs").as_slice(),
        ),
        (
            b"ghost_tile_kernel.cuh".as_slice(),
            include_bytes!("cuda/ghost_tile_kernel.cuh").as_slice(),
        ),
        (
            b"ghost_tile_kernel.cu".as_slice(),
            include_bytes!("cuda/ghost_tile_kernel.cu").as_slice(),
        ),
        (
            b"captured_pipeline.rs".as_slice(),
            include_bytes!("captured_pipeline.rs").as_slice(),
        ),
    ] {
        hasher.update((label.len() as u64).to_le_bytes());
        hasher.update(label);
        hasher.update((source.len() as u64).to_le_bytes());
        hasher.update(source);
    }
    ghost_sha256_hex(&hasher.finalize())
}

fn ghost_sha256_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// One exact residue/spike-count row decoded from a v3 record.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GhostClusterResidueSpikeCount {
    pub residue_id: u32,
    pub spike_count: u32,
}

/// Strict v3 RichSpike cluster-residue membership payload. It is intentionally
/// not named or interpreted as a cavity-lining/contact set. Depth, SASA and
/// coordinate-derived structural response remain unavailable until a
/// hash-bound coordinate-frame sidecar is joined; they are never inferred
/// from the AABB midpoint.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GhostV3ClusterResidueMembership {
    pub membership_semantics: String,
    pub source_spike_count: u32,
    pub unresolved_residue_spike_count: u32,
    pub cluster_residue_spike_counts: Vec<GhostClusterResidueSpikeCount>,
    pub coordinate_frame_status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coordinate_frame_sha256: Option<String>,
    pub depth_status: String,
    pub sasa_status: String,
    pub structural_response_status: String,
}

/// Decode and validate a production v3 cluster-residue membership payload.
/// Any truncation or overflow is fatal because a partial spike population
/// would change cluster identity and invalidate Path-B/fiber joins.
pub fn decode_ghost_v3_cluster_residue_membership(
    record: &[u8],
) -> Result<GhostV3ClusterResidueMembership, String> {
    if record.len() != GHOST_RECORD_BYTES {
        return Err(format!(
            "Ghost v3 record length {} != {}",
            record.len(),
            GHOST_RECORD_BYTES
        ));
    }
    let read_u32 = |offset: usize| -> u32 {
        u32::from_le_bytes(
            record[offset..offset + 4]
                .try_into()
                .expect("fixed v3 field"),
        )
    };
    if read_u32(GHOST_V2_OFFSET_SCHEMA_VERSION) != GHOST_FRAME_SCHEMA_V3
        || read_u32(GHOST_V3_OFFSET_MEMBERSHIP_SCHEMA_VERSION) != GHOST_V3_MEMBERSHIP_SCHEMA_VERSION
    {
        return Err(
            "Ghost record is not production v3 cluster-residue membership schema".to_string(),
        );
    }
    let count = read_u32(GHOST_V3_OFFSET_CLUSTER_RESIDUE_COUNT) as usize;
    let capacity = read_u32(GHOST_V3_OFFSET_CLUSTER_RESIDUE_CAPACITY) as usize;
    let overflow = read_u32(GHOST_V3_OFFSET_CLUSTER_RESIDUE_OVERFLOW);
    let source_spike_count = read_u32(GHOST_V3_OFFSET_SOURCE_SPIKE_COUNT);
    let unresolved = read_u32(GHOST_V3_OFFSET_UNRESOLVED_RESIDUE_SPIKE_COUNT);
    let entry_bytes = read_u32(GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRY_BYTES) as usize;
    let availability = read_u32(GHOST_V3_OFFSET_GEOMETRY_AVAILABILITY_FLAGS);
    if capacity != GHOST_V3_CLUSTER_RESIDUE_CAPACITY
        || entry_bytes != GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES
        || count > capacity
        || overflow != 0
        || availability & GHOST_V3_GEOMETRY_CLUSTER_RESIDUE_MEMBERSHIP_EXACT == 0
        || availability & GHOST_V3_GEOMETRY_COORDINATE_SIDECAR_REQUIRED == 0
    {
        return Err(format!(
            "Ghost v3 cluster-residue membership contract failed: count={} capacity={} entry_bytes={} overflow={} flags=0x{:08x}",
            count, capacity, entry_bytes, overflow, availability
        ));
    }
    let mut memberships = Vec::with_capacity(count);
    for index in 0..count {
        let offset =
            GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES + index * GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES;
        let residue_id = read_u32(offset);
        let spike_count = read_u32(offset + 4);
        if spike_count == 0 {
            return Err(format!(
                "Ghost v3 cluster-residue membership {} has zero spike count",
                index
            ));
        }
        memberships.push(GhostClusterResidueSpikeCount {
            residue_id,
            spike_count,
        });
    }
    memberships.sort_by_key(|membership| membership.residue_id);
    if memberships
        .windows(2)
        .any(|pair| pair[0].residue_id == pair[1].residue_id)
    {
        return Err(
            "Ghost v3 cluster-residue membership contains duplicate residue identities".to_string(),
        );
    }
    if memberships.is_empty() {
        return Err(
            "Ghost v3 record has no resolved cluster-residue membership and cannot form a cluster identity"
                .to_string()
        );
    }
    let resolved_total = memberships.iter().try_fold(0u32, |sum, membership| {
        sum.checked_add(membership.spike_count)
            .ok_or_else(|| "Ghost v3 cluster-residue spike-count sum overflow".to_string())
    })?;
    if resolved_total.checked_add(unresolved) != Some(source_spike_count) {
        return Err(format!(
            "Ghost v3 source/membership conservation failed: source={} resolved={} unresolved={}",
            source_spike_count, resolved_total, unresolved
        ));
    }
    let coordinate_sha = &record
        [GHOST_V3_OFFSET_COORDINATE_FRAME_SHA256..GHOST_V3_OFFSET_COORDINATE_FRAME_SHA256 + 32];
    if coordinate_sha.iter().any(|byte| *byte != 0) {
        return Err(
            "Ghost v3 record carries an unverified inline coordinate-frame digest".to_string(),
        );
    }
    Ok(GhostV3ClusterResidueMembership {
        membership_semantics: "rich_spike_cluster_membership_not_cavity_lining".to_string(),
        source_spike_count,
        unresolved_residue_spike_count: unresolved,
        cluster_residue_spike_counts: memberships,
        coordinate_frame_status: "unavailable_content_addressed_coordinate_frame_sidecar_required"
            .to_string(),
        coordinate_frame_sha256: None,
        depth_status: "unavailable_requires_coordinate_frame_sidecar".to_string(),
        sasa_status: "unavailable_requires_coordinate_frame_sidecar".to_string(),
        structural_response_status: "unavailable_requires_matched_coordinate_frame_sidecars"
            .to_string(),
    })
}

/// **GHOST_NATIVE_SPATIAL_MAPPING_WIRE — field_completeness_flags bits.**
///
/// Bits 0-3 are populated by the v2 kernel directly (kl-finite, gear, dt,
/// not-tainted). Bit 1 is set only when `gear_id` is both non-zero AND not
/// the GEAR-PROVENANCE-002 "not measured" sentinel `0xFFFFFFFF`
/// (`cuda/ghost_tile_kernel.cu:618-625`; the Rust-side value is
/// `crate::zstr::ZSTR_GEAR_ID_UNMEASURED`, `zstr.rs:569` — this module
/// defines no mirror of the CUDA `GHOST_V2_GEAR_ID_UNMEASURED`). The v2
/// `gear_id` is a kernel-node argument fixed at graph build while the gear
/// is chosen per replay by the device SFA, so the production launcher always
/// stamps the sentinel (`captured_pipeline.rs:4650-4659`) and Bit 1 is
/// correspondingly CLEAR on the production path. A set Bit 1 therefore means
/// a measured gear, never merely a non-zero word; readers fall back to
/// `dt_fs`.
///
/// Bit 4 is set when AABB / centroid spatial fields at
/// offsets 160 / 172 / 184 were natively populated from
/// `ContactShellTile.aabb_min/aabb_max` at write time. When Bit 4 == 0,
/// the spatial fields are sentinel-zero and must be treated as missing
/// by any downstream consumer; the host audit (probe / scanner) reads
/// this bit to distinguish native-populated from sentinel-zero records.
///
/// The selected centroid view at offset 184 is the AABB midpoint —
/// labelled as `aabb_midpoint_native_contact_shell_tile`. It is NOT a
/// phase-manifold complete centroid; richer views (spike_density,
/// kcc_driver, phasor_coherent, thermo_weighted, ghost_zstr_event_weighted)
/// are computed offline by the SiteManifest materializer.
pub const GHOST_FCF_BIT_KL_FINITE: u16 = 0x0001;
/// NAME IS A MISNOMER for the semantics it guards: the producer excludes the
/// not-measured sentinel as well as zero, so this bit means "measured gear",
/// not "gear_id != 0" — see the sentinel exclusion documented above.
/// Renaming the constant is an API change and is out of scope here.
pub const GHOST_FCF_BIT_GEAR_ID_NONZERO: u16 = 0x0002;
pub const GHOST_FCF_BIT_DT_FS_POSITIVE: u16 = 0x0004;
pub const GHOST_FCF_BIT_THERMO_NOT_TAINTED: u16 = 0x0008;
pub const GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT: u16 = 0x0010;

// The v2 region MUST end at or before _slack (offset 256).
const _: () = {
    assert!(GHOST_V2_OFFSET_SCHEMA_VERSION >= 128);
    assert!(GHOST_V2_OFFSET_OBSERVATION_PASS == 132);
    assert!(GHOST_V2_OFFSET_DISCOVERY_PASS == 133);
    assert!(GHOST_V2_OFFSET_PERTURBATION_CHAN == 134);
    assert!(GHOST_V2_OFFSET_UV_WAVELENGTH_NM == 136);
    assert!(GHOST_V2_OFFSET_GEAR_ID == 140);
    assert!(GHOST_V2_OFFSET_DT_FS == 144);
    assert!(GHOST_V2_OFFSET_STEP_IDX == 152);
    assert!(GHOST_V2_OFFSET_STEP_IDX % 8 == 0); // 8-aligned for STG.E.64
    assert!(GHOST_V2_OFFSET_AABB_MIN == 160);
    assert!(GHOST_V2_OFFSET_AABB_MAX == 172);
    assert!(GHOST_V2_OFFSET_CENTROID == 184);
    assert!(GHOST_V2_OFFSET_V2_RESERVED == 196);
    // v2 fields end at: 196 + 60 = 256 — must equal _slack offset.
    assert!(GHOST_V2_OFFSET_V2_RESERVED + 60 == 256);
    // Reserved-payload region begins at 128 and ends just before _slack at 256.
    assert!(GHOST_V2_OFFSET_SCHEMA_VERSION + 128 == 256);
    assert!(GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES == 320);
    assert!(GHOST_V3_CLUSTER_RESIDUE_CAPACITY == 472);
    assert!(
        GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES
            + GHOST_V3_CLUSTER_RESIDUE_CAPACITY * GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES
            == GHOST_RECORD_BYTES
    );

    // ── M1.2.24 — natural-alignment audit ─────────────────────────────────
    // Every offset must be naturally aligned for its load/store width on
    // CUDA Blackwell (sm_120 issues STG.E.64/STG.E.32/STG.E.16 which trap
    // CUDA_ERROR_MISALIGNED_ADDRESS on misaligned destinations). Prior
    // step_idx misalignment at offset 148 was the M1.2.24 root cause.
    // This block is regression protection.
    assert!(GHOST_V2_OFFSET_SCHEMA_VERSION % 4 == 0); // u32
    // The three `% 1 == 0` predicates below are true for every usize, so they
    // pin nothing. The real u8 offsets are pinned by the `== 132`, `== 133`
    // and `== 134` asserts in the block above; replacing these tautologies
    // with those values would change executable code and is out of scope for
    // a comment-only phase.
    assert!(GHOST_V2_OFFSET_OBSERVATION_PASS % 1 == 0); // u8 — trivially aligned
    assert!(GHOST_V2_OFFSET_DISCOVERY_PASS % 1 == 0); // u8 — trivially aligned
    assert!(GHOST_V2_OFFSET_PERTURBATION_CHAN % 1 == 0); // u8 — trivially aligned
    assert!(GHOST_V2_OFFSET_UV_WAVELENGTH_NM % 2 == 0); // u16
    assert!(GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS % 2 == 0); // u16
    assert!(GHOST_V2_OFFSET_GEAR_ID % 4 == 0); // u32
    assert!(GHOST_V2_OFFSET_DT_FS % 4 == 0); // f32
    assert!(GHOST_V2_OFFSET_STEP_IDX % 8 == 0); // u64 ← M1.2.24 fix
    assert!(GHOST_V2_OFFSET_AABB_MIN % 4 == 0); // f32[3] component-wise
    assert!(GHOST_V2_OFFSET_AABB_MAX % 4 == 0); // f32[3]
    assert!(GHOST_V2_OFFSET_CENTROID % 4 == 0); // f32[3]
    assert!(GHOST_V2_OFFSET_V2_RESERVED % 4 == 0); // u32 array
};

/// UV wavelength bit-code → wavelength_nm, ordered to match `MU_01_SQ_LUT`
/// (`crates/prism-nhs/src/interferometric_adjudicator.rs:742` `QI_SHIFT` = 30,
/// `:752` `MU_01_SQ_LUT`).
///
/// Not currently exercised by any producer: the v2 kernel writes
/// `uv_wavelength_nm = 0` and `perturbation_channel = UNKNOWN`
/// (`cuda/ghost_tile_kernel.cu:612-613`), so no record carries a decodable
/// bitcode, and no consumer in this tree reads this table — its only
/// occurrence is this definition. It is the decode contract for a
/// perturbation-channel wire-up that has not landed.
pub const GHOST_UV_WAVELENGTH_NM_BY_BITCODE: [u16; 4] = [260, 280, 305, 320];

/// Sentinel for "perturbation channel unknown / not applicable to this record".
pub const GHOST_PERTURBATION_CHANNEL_UNKNOWN: u8 = 0xFF;

/// Read schema_version from a 4096-byte record. Returns the raw u32.
/// Production records carry `GHOST_FRAME_SCHEMA_V3` = 3 — that is the only
/// value any producer in this tree writes into this field
/// (`cuda/ghost_tile_kernel.cu:609`). `GHOST_FRAME_SCHEMA_V1_LEGACY` = 0
/// identifies a legacy record whose reserved region was only zeroed.
/// `GHOST_FRAME_SCHEMA_V2` = 2 is defined but written by no producer, and 1
/// is reserved and never used, so a 1 reading is drift or corruption.
/// A consumer gating on the production stream MUST match V3: matching only
/// V1/V2 rejects every record the kernel emits.
#[inline]
pub fn ghost_v2_read_schema_version(record_bytes: &[u8; GHOST_RECORD_BYTES]) -> u32 {
    let off = GHOST_V2_OFFSET_SCHEMA_VERSION;
    let mut b = [0u8; 4];
    b.copy_from_slice(&record_bytes[off..off + 4]);
    u32::from_le_bytes(b)
}

// ─── GhostTileRing — pinned-host, device-mapped ring buffer ─────────────────

/// Explicitly initialize every byte in a newly allocated Ghost staging ring.
/// CUDA does not promise zero-filled pinned host allocations, while legacy
/// schema-0 producers intentionally leave record regions unwritten.  The
/// caller must supply the complete allocation span before any device alias is
/// published or any producer can observe the allocation.
unsafe fn zero_ghost_ring_allocation(host_base: *mut u8, total_bytes: usize) -> Result<(), String> {
    if host_base.is_null() {
        return Err("cannot initialize a null Ghost staging allocation".to_string());
    }
    if total_bytes == 0 {
        return Err("cannot initialize an empty Ghost staging allocation".to_string());
    }
    unsafe {
        std::ptr::write_bytes(host_base, 0u8, total_bytes);
    }
    Ok(())
}

/// Pinned-host buffer with a device-mapped alias.  The captured kernel
/// writes records through the device pointer; the host reads through the
/// host pointer with no DtoH copy.
///
/// Layout in the buffer (v9D' / Amendment 3.14):
/// ```text
///   [u32 n_frames_written]         offset 0..4
///   [u8  _pad[4092]]               offset 4..4096 (counter sector)
///   [GhostTileFrame] × max_records offset 4096..end (4096 B per record)
/// ```
///
/// The 4096-byte leading counter sector reserves space for the atomic
/// `n_frames_written` counter that the GPU kernel atomicAdds and the
/// host reads on teardown.  Aligning the first record to offset 4096
/// means record N begins at ring byte 4096·(N+1), so every record offset
/// is an exact sector multiple and no partial-sector arithmetic is
/// needed.  (This ring is pinned host memory; the durable write path is
/// buffered + `sync_all` + rename, not `O_DIRECT` — see the module doc.)
#[cfg(feature = "gpu")]
pub struct GhostTileRing {
    /// Pinned-host base pointer (page-locked, mapped). The first 4096 B
    /// hold the `n_frames_written` u32 counter + padding; records start at
    /// the sector-aligned offset 4096 used by `COUNTER_SECTOR_BYTES`.
    pub host_base: *mut u8,
    /// Device-mapped alias of `host_base` — kernels write through this.
    pub device_base: u64,
    /// Total bytes allocated (counter sector + max_records * 4096).
    pub total_bytes: usize,
    /// Maximum number of records the buffer can hold without overflow.
    /// The kernel claims a slot atomically FIRST, then bounds-checks the
    /// claimed index before writing and compensates the counter on
    /// overflow — `atomicAdd` then `atomicMin` in the v1 kernel
    /// (`cuda/ghost_tile_kernel.cu:138-145`), `atomicAdd` then
    /// `atomicSub` in the v2 kernel (`:542-546`).
    pub max_records: u32,
}

#[cfg(feature = "gpu")]
unsafe impl Send for GhostTileRing {}
#[cfg(feature = "gpu")]
unsafe impl Sync for GhostTileRing {}

#[cfg(feature = "gpu")]
impl GhostTileRing {
    /// Reserved counter-sector size at the head of the buffer.  The
    /// first record starts at offset `COUNTER_SECTOR_BYTES`.  Sized to
    /// 4096 B in v9D' so the counter sector and the first record both
    /// land on aligned NVMe physical sectors when the buffer base is
    /// page-aligned (cuMemHostAlloc returns page-aligned pointers).
    pub const COUNTER_SECTOR_BYTES: usize = 4096;

    /// Allocate a pinned-host, device-mapped ring sized for
    /// `max_records` records of 4096 bytes each. Every byte is explicitly
    /// zeroed once before the device alias is published; allocation behavior
    /// is never treated as initialization evidence.
    pub fn allocate(max_records: u32) -> Result<Self, String> {
        let total_bytes = Self::COUNTER_SECTOR_BYTES + (max_records as usize) * GHOST_RECORD_BYTES;

        let mut host_ptr: *mut c_void = std::ptr::null_mut();
        let flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
        let rc = unsafe { cuMemHostAlloc(&mut host_ptr, total_bytes, flags) };
        if !matches!(rc, CUresult::CUDA_SUCCESS) {
            return Err(format!(
                "cuMemHostAlloc(GhostTileRing, {} B) failed: rc={:?}",
                total_bytes, rc
            ));
        }
        if host_ptr.is_null() {
            return Err("cuMemHostAlloc returned null".to_string());
        }

        // Legacy schema-0 producers do not populate every record byte. CUDA
        // does not guarantee zero-filled pinned allocations, so initialize the
        // counter sector and the complete record/slack population exactly once
        // before publishing a device-mapped alias.
        if let Err(error) = unsafe { zero_ghost_ring_allocation(host_ptr as *mut u8, total_bytes) }
        {
            unsafe {
                let _ = cuMemFreeHost(host_ptr);
            }
            return Err(error);
        }

        // Get the device-side mapped alias.  On Blackwell + PCIe Gen5
        // this is the same VA window as the host pointer, so memcpy is
        // not required — kernel STG.E.* lands directly on host RAM.
        let mut dev_ptr: u64 = 0;
        let rc = unsafe { cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0) };
        if !matches!(rc, CUresult::CUDA_SUCCESS) {
            unsafe {
                let _ = cuMemFreeHost(host_ptr);
            }
            return Err(format!("cuMemHostGetDevicePointer_v2 failed: rc={:?}", rc));
        }

        Ok(Self {
            host_base: host_ptr as *mut u8,
            device_base: dev_ptr,
            total_bytes,
            max_records,
        })
    }

    /// Read the GPU-written `n_frames_written` counter.  Volatile so the
    /// load isn't hoisted past the post-kernel synchronize.  Returns
    /// the count of 4096-byte `GhostTileFrame` records currently resident
    /// in the buffer.  (The two-part 1408-byte header+ContactShellTile
    /// record was retired by Amendment 3.14 — see `GHOST_RECORD_BYTES`.)
    pub fn n_frames_written(&self) -> u32 {
        unsafe { std::ptr::read_volatile(self.host_base as *const u32) }
    }

    /// Byte slice of the live record payload (excluding the leading
    /// counter sector).  Length is clamped to `min(n_frames_written,
    /// max_records) * 4096`.
    pub fn payload_bytes(&self) -> &[u8] {
        let n = self.n_frames_written().min(self.max_records);
        let nbytes = (n as usize) * GHOST_RECORD_BYTES;
        unsafe {
            std::slice::from_raw_parts(self.host_base.add(Self::COUNTER_SECTOR_BYTES), nbytes)
        }
    }

    /// Persist every currently produced record and reset the staging ring.
    ///
    /// # Safety
    ///
    /// The caller must synchronize the CUDA stream that owns every kernel
    /// capable of writing this ring before calling. No such kernel may be
    /// launched until this function returns. This exclusive boundary is what
    /// makes segment rollover exactly-once: bytes are fsynced and indexed
    /// before the producer counter is reset to zero.
    pub unsafe fn persist_and_reset_after_stream_sync(
        &self,
        writer: &mut crate::ghost_segment::GhostSegmentWriter,
    ) -> anyhow::Result<u64> {
        let overflow = &writer.manifest().overflow_conservation;
        if overflow.admission_status != "BOUND_ENGINE_RUNTIME"
            || overflow.ring_capacity_records != self.max_records as u64
            || !overflow.replay_upper_bound_admission_pass
        {
            anyhow::bail!(
                "Ghost staging drain is not bound to this ring capacity: ring_capacity={} authority_capacity={} status={}",
                self.max_records,
                overflow.ring_capacity_records,
                overflow.admission_status,
            );
        }
        let produced = self.n_frames_written() as u64;
        if produced > self.max_records as u64 {
            anyhow::bail!(
                "Ghost staging ring overflow: produced={} capacity={}",
                produced,
                self.max_records
            );
        }
        writer.append_payload(self.payload_bytes(), produced)?;
        // The release fence prevents the counter reset from becoming visible
        // before the durable writer has committed its segment + manifest.
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        std::ptr::write_volatile(self.host_base as *mut u32, 0u32);
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        Ok(produced)
    }
}

#[cfg(feature = "gpu")]
impl Drop for GhostTileRing {
    fn drop(&mut self) {
        if !self.host_base.is_null() {
            unsafe {
                let _ = cuMemFreeHost(self.host_base as *mut c_void);
            }
            self.host_base = std::ptr::null_mut();
        }
    }
}

// ─── FFI declarations for the C-side capture kernel ─────────────────────────

#[cfg(feature = "gpu")]
extern "C" {
    /// Legacy captured-graph kernel declaration, still reachable when
    /// `cfg.mar_v2` is absent — that is the `else` arm of the launcher
    /// selection at `captured_pipeline.rs:4637`, which calls this entry at
    /// `:4858`. The production path is the v2 launcher, which emits one v3
    /// Ghost record with cluster-residue membership whenever the v2
    /// emission gate passes (`cuda/ghost_tile_kernel.cu:534-538`), not only
    /// on `adj.adjudication_code >= 1`.
    ///
    /// `ring_base_dev`: device-mapped pointer (from `GhostTileRing::device_base`)
    /// `tiles`:         baseline manifold pointer (n_clusters × ContactShellTile)
    /// `adj`:           adjudicator FFI struct
    /// `frame_idx`:     monotonic frame counter (host-supplied per launch)
    /// `n_clusters`:    cluster count
    /// `max_records`:   bounds check — kernel skips writes once counter >= this
    /// Reconciled with the canonical declaration in
    /// `captured_pipeline.rs:802` — adds the `firehose_enable: u32` final
    /// argument that matches the actual CUDA `extern "C"` signature in
    /// `crates/prism-nhs/src/cuda/ghost_tile_kernel.cu:267-277`. The
    /// previous 8-argument declaration triggered
    /// `clashing_extern_declarations` warnings at every full-tree compile.
    pub fn prism_ghost_pipe_stage_launch(
        ring_base_dev: u64,
        tiles: *const std::ffi::c_void,
        adj: *const std::ffi::c_void,
        d_kcc_lead: *const std::ffi::c_void, // Wave 1 / Q2 — nullable
        frame_idx: u64,
        n_clusters: u32,
        max_records: u32,
        stream: *mut std::ffi::c_void,
        firehose_enable: u32,
    ) -> i32;

    /// Wave 1 / Q1 — populator for the __constant__ d_cluster_to_repr_residue[64]
    /// table.  Stream-ordered cudaMemcpyToSymbolAsync.  `n` clamped to 64.
    pub fn prism_ghost_set_cluster_repr_residue(
        repr_residues_host: *const u32,
        n: u32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// **M1.2.20.C-C / T20 + Amendment 3.20** — Topology-Driven Chain
    /// Boundary populator.  Host parses topology.chain_ids[] and
    /// produces an 8-entry residue-boundary array where
    /// `offsets_host[i]` is the first residue id of chain `i`, with the
    /// unused slots set to `u32::MAX` by the caller (every production
    /// call site passes all 8: `bin/nhs_rt_full.rs:18973`,
    /// `coupled_md.rs:4897` and `:9066`).
    /// Stream-ordered cudaMemcpyToSymbolAsync; `n` clamped to 8.
    /// Target-agnostic — replaces all prior hardcoded chain numbers.
    pub fn prism_ghost_set_chain_offsets(
        offsets_host: *const u32,
        n: u32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// **Path Ω Phase 2** — geometry-emergent chain_id LUT populator.
    /// `chain_ids_host[i]` ∈ {0, 1, 0xFF} for cluster i.  Computed
    /// host-side from cluster centroids vs dyad axis at V2 build.
    /// Sentinel 0xFF disables the geometry-emergent path for that
    /// slot (kernel falls back to residue-id boundary scan).
    pub fn prism_ghost_set_cluster_chain_id(
        chain_ids_host: *const u8,
        n: u32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

/// **Path Ω Phase 2** — Public host wrapper for the geometry-emergent
/// chain_id LUT populator.
///
/// # Safety
/// `chain_ids_host` must point to at least `n` valid `u8` values.
/// `stream` must be a valid CUstream owned by the active context.
#[cfg(feature = "gpu")]
pub unsafe fn set_cluster_chain_id(
    chain_ids_host: *const u8,
    n: u32,
    stream: *mut std::ffi::c_void,
) -> i32 {
    prism_ghost_set_cluster_chain_id(chain_ids_host, n, stream)
}

/// **M1.2.20.C-C / T20 + Amendment 3.20** — Public host wrapper for
/// the topology-driven chain offsets populator.  The underlying
/// `prism_ghost_set_chain_offsets` extern is declared at module-level
/// (not in a sub-module), so this just provides a documented Rust
/// signature for the bin call site.
///
/// # Safety
/// `offsets_host` must point to at least `n` valid `u32` values.
/// `stream` must be a valid CUstream owned by the active context.
#[cfg(feature = "gpu")]
pub unsafe fn set_chain_offsets(
    offsets_host: *const u32,
    n: u32,
    stream: *mut std::ffi::c_void,
) -> i32 {
    prism_ghost_set_chain_offsets(offsets_host, n, stream)
}

/// **Wave 1 / Q1 — TIER 1 leak-fix (2026-05-03)** — Public host wrapper
/// for the cluster→representative-residue LUT populator.  Loads the
/// __constant__ d_cluster_to_repr_residue[64] slab consumed by the
/// ghost-tile kernel's CHAIN-ID resolution path: when the Path-Ω
/// geometry-emergent `d_cluster_to_chain_id[i]` slot holds the 0xFF
/// "unset" sentinel, the kernel scans this cluster's representative
/// residue against `d_chain_offsets` to assign `chain_id`
/// (`cuda/ghost_tile_kernel.cu:175-188` and `:556-569` — the only two
/// reads of the slab). That branch is LIVE, not vestigial: it is the
/// branch taken whenever the geometry-emergent LUT is unpopulated,
/// which is the normal state for a single-chain target. This
/// representative-residue telemetry is not a cavity-lining assignment.
/// Without the LUT, every cluster slot reads the kernel default (zero),
/// so every boundary scan runs against residue 0 and chain attribution
/// collapses onto chain 0.
///
/// (That collapse holds while `d_chain_offsets` carries its `u32::MAX`
/// padding — the fold at `cuda/ghost_tile_kernel.cu:181-186` and `:562-567`
/// then matches only boundary 0. If `d_chain_offsets` is left unpopulated
/// as well it reads as all-zero `__constant__` memory, every `k ∈ 0..8`
/// satisfies `boundary != 0xFFFFFFFF && boundary <= repr_res`, and the fold
/// terminates at `chain_id = 7` — a chain index no topology defines. Both
/// outcomes presuppose the fold is entered at all, i.e.
/// `d_cluster_to_chain_id[i] == 0xFF`; when that Path-Ω LUT is itself
/// unpopulated it reads 0 and every record is stamped chain 0 without the
/// boundary scan ever running.)
///
/// This LUT is NOT a fallback for `causal_lead_residue`, and no such
/// fallback exists anywhere in the kernels. When the F2-pool
/// `d_kcc_lead` buffer is absent the kernel writes the 0xFFFFFFFF
/// sentinel verbatim (`cuda/ghost_tile_kernel.cu:231-233` and `:600`).
/// That sentinel is the correct and intended behaviour and MUST NOT be
/// "repaired" by substituting the representative residue. The host
/// computes this LUT as the MODE of `residue_id` over the cluster's
/// spike range (`bin/nhs_rt_full.rs:20807-20820`) — a population
/// statistic describing which residue contributes the most spikes.
/// `causal_lead_residue` is a different, measured quantity: the
/// per-cluster `argmax|d_kcc_temporal_corr|` over the KCC causal plane
/// (`captured_pipeline.rs:1568-1569`). Writing the mode into that field
/// would publish a population statistic under the name of a measured
/// causal driver — fabricated evidence that no downstream Lag-Persistence
/// consumer could tell apart from a real KCC measurement. Leave the
/// sentinel.
///
/// # Safety
/// `repr_residues_host` must point to at least `n` valid `u32` values.
/// `n` is clamped to 64 inside the FFI; values past 64 are silently
/// dropped.  `stream` must be a valid CUstream owned by the active
/// context.
#[cfg(feature = "gpu")]
pub unsafe fn set_cluster_repr_residue(
    repr_residues_host: *const u32,
    n: u32,
    stream: *mut std::ffi::c_void,
) -> i32 {
    prism_ghost_set_cluster_repr_residue(repr_residues_host, n, stream)
}

#[cfg(test)]
mod allocation_initialization_tests {
    use super::*;

    #[test]
    fn zero_initialization_covers_counter_records_and_legacy_slack() {
        let counter_sector_bytes = GHOST_RECORD_BYTES;
        let total_bytes = counter_sector_bytes + 2 * GHOST_RECORD_BYTES;
        let mut allocation = vec![0xa5u8; total_bytes];

        unsafe {
            zero_ghost_ring_allocation(allocation.as_mut_ptr(), allocation.len()).unwrap();
        }

        assert!(allocation.iter().all(|byte| *byte == 0));
        assert_eq!(allocation[0], 0);
        assert_eq!(allocation[counter_sector_bytes], 0);
        assert_eq!(
            allocation[counter_sector_bytes + GHOST_V2_OFFSET_SCHEMA_VERSION],
            0
        );
        assert_eq!(
            allocation[counter_sector_bytes + GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES],
            0
        );
        assert_eq!(allocation[total_bytes - 1], 0);
    }

    #[test]
    fn zero_initialization_rejects_invalid_spans() {
        let mut byte = 0xa5u8;
        assert!(unsafe { zero_ghost_ring_allocation(std::ptr::null_mut(), 1) }.is_err());
        assert!(unsafe { zero_ghost_ring_allocation(&mut byte, 0) }.is_err());
        assert_eq!(byte, 0xa5);
    }
}

#[cfg(test)]
mod plane_evidence_tests {
    use super::*;
    #[test]
    fn padding_is_not_occupancy_or_cosine_evidence() {
        let mut record=vec![0u8;4096];
        for i in 0..24 {record[20+4*i..24+4*i].copy_from_slice(&1e-7f32.to_le_bytes());}
        let unknown=ghost_plane_evidence(&record).unwrap();
        assert_eq!(unknown.producer_populated_mask,None);
        assert_eq!(unknown.exact_floor_only_mask,15);
        assert_eq!(unknown.cosine_evidence_mask,0);
        record[20..24].copy_from_slice(&0.25f32.to_le_bytes());
        assert_eq!(ghost_plane_evidence(&record).unwrap().cosine_evidence_mask,1);
        record[24..28].copy_from_slice(&0.0f32.to_le_bytes());
        assert_eq!(ghost_plane_evidence(&record).unwrap().cosine_evidence_mask,1);
        assert_eq!(ghost_plane_evidence(&record).unwrap().below_known_padding_floor_mask,1);
        record[196..200].copy_from_slice(&(GHOST_PLANE_MASK_PRESENT|15).to_le_bytes());
        let declared=ghost_plane_evidence(&record).unwrap();
        assert_eq!(declared.producer_populated_mask,Some(GHOST_PLANE_MASK_PRESENT|15));
        assert_eq!(declared.cosine_evidence_mask,1);
        record[20..24].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(ghost_plane_evidence(&record).unwrap().cosine_evidence_mask,0);
    }
}
