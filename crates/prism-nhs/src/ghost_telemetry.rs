//! CLA-2 — Ghost Telemetry Pipeline.
//!
//! Per the Anti-Greenfield Doctrine + Host-Sync-Fallacy mandate
//! (operator directive 2026-04-29 Part 2). Asynchronous, transparent
//! exfiltration of `ContactShellTile` data (or any other POD payload)
//! from VRAM into a triple-buffered, pinned host ring on a dedicated
//! non-blocking telemetry stream. The host is now a "secretary" that
//! logs reports the GPU sends to the ghost buffer; it never gates the
//! in-flight physics or the ASC steering decisions.
//!
//! # Why this module exists (Anti-Greenfield justification §J)
//!
//! `vram_pool.rs` wraps the F2 VRAM-only allocator. `so3_project.rs`
//! owns the `ContactShellTile` schema + the SO(3) projection
//! transform. Neither is the right home for the host-pinned + async-DMA
//! orchestration: that surface is its own concern (host memory class +
//! stream-priority + ring rotation). This is the single new module
//! Task #24 (CLA-2) requires; everything else is scavenged.
//!
//! # Hardware-Physics Contract (operator §2)
//!
//! 1. **Non-Blocking stream**: every DMA is launched on a stream
//!    created with `CU_STREAM_NON_BLOCKING`. Default-stream (Stream 0)
//!    operations would implicitly synchronise with the main MD
//!    integrator stream, recreating the host-sync fallacy we are
//!    eliminating.
//!
//! 2. **Pinned host buffer**: `cuMemAllocHost_v2` produces page-locked
//!    host memory so the DMA engine can stream over PCIe Gen5 at
//!    ~64 GB/s without page-fault stalls (10–100 ms vs sub-µs
//!    headroom).
//!
//! 3. **Triple-buffered ring**: `[T; 3]` rotation indexed by
//!    `frame_idx % 3`. A double-buffer is insufficient under burst
//!    spike-rates where the host JSON serializer can lag the device
//!    write; the third slot guarantees the safe-to-read slot is
//!    never overwritten by an in-flight DMA.
//!
//! Slot life-cycle, with `i = frame_idx`:
//!
//! | slot index | role                                  |
//! |---|---|
//! | `i       % 3` | being written by the current DMA       |
//! | `(i - 1) % 3` | purgatory (transfer may be in flight)  |
//! | `(i - 2) % 3` | safe-to-read for host-side serializer  |
//!
//! # Cross-lane FFI surface
//!
//! The ring's `write_slot_dst_ptr(frame_idx)` returns a `*mut c_void`
//! that the captured CUDA graph passes to `cuMemcpyDtoHAsync_v2` as
//! the destination. The Adjudicator (Claude-2) does not consume this
//! ring — it reads `tiles_dev_ptr` directly from the
//! [`crate::so3_project::SiteManifestFfi`] in VRAM. The ring is
//! exclusively the host's "ghost-side" view of that same data,
//! intentionally lagged by ≥ 2 frames.

#![cfg(feature = "gpu")]

use cudarc::driver::sys::{
    cuMemAllocHost_v2, cuMemFreeHost, cuMemcpyDtoHAsync_v2, cuPointerGetAttribute, cuStreamCreate,
    cuStreamGetFlags, CUdeviceptr, CUmemorytype_enum, CUpointer_attribute_enum, CUresult, CUstream,
    CUstream_flags_enum,
};
use std::ffi::c_void;
use std::marker::PhantomData;

// ============================================================================
// PinnedTelemetryRing<T> — triple-buffered pinned host ring
// ============================================================================

/// Triple-buffered pinned-host ring for asynchronous device → host
/// exfiltration of POD elements of type `T`.
///
/// Slot count is fixed at 3 (per the operator's mandate §2.3). Each
/// slot holds `elems_per_slot` elements; the ring's total pinned
/// allocation is `3 * elems_per_slot * size_of::<T>()` bytes.
///
/// # Safety contract on `T`
///
/// `T` MUST be POD (no Drop, no interior pointers that need
/// host-side dereferencing through the ring). The ring provides a
/// `&[T]` view of the read slot via [`Self::read_slot_unchecked`]
/// which reads memory the GPU asynchronously wrote — the caller
/// must have synchronized the telemetry stream (or the relevant
/// CUDA event) before treating the slice as up-to-date.
///
/// # Layout invariant
///
/// The pinned allocation is contiguous: slot `i` starts at
/// `base + i * elems_per_slot * size_of::<T>()`. No padding between
/// slots; CUDA pinned-memory allocations are page-aligned by
/// construction so the slot stride is also DMA-friendly.
pub struct PinnedTelemetryRing<T> {
    base: *mut T,
    elems_per_slot: usize,
    _marker: PhantomData<T>,
}

impl<T> PinnedTelemetryRing<T> {
    /// Number of slots in the ring. Fixed at 3 (operator mandate §2.3).
    pub const N_SLOTS: usize = 3;

    /// Allocate a new pinned-host ring with `elems_per_slot`
    /// elements per slot. Returns `Err(cudaError)` if the
    /// `cuMemAllocHost_v2` call fails.
    ///
    /// `elems_per_slot == 0` is allowed and produces a zero-byte
    /// pinned allocation; useful for the "no clusters yet" branch
    /// of an MD-campaign init that grows the ring on demand.
    pub fn new(elems_per_slot: usize) -> Result<Self, i32> {
        let total_elems = Self::N_SLOTS * elems_per_slot;
        let total_bytes = total_elems
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE as i32)?;

        let mut p: *mut c_void = std::ptr::null_mut();
        if total_bytes > 0 {
            let rc = unsafe { cuMemAllocHost_v2(&mut p as *mut _, total_bytes) };
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                return Err(rc as i32);
            }
            if p.is_null() {
                return Err(CUresult::CUDA_ERROR_OUT_OF_MEMORY as i32);
            }
            // Initialize the entire owned ring before any captured DMA. This
            // makes untouched rotation slots defined checkpoint state.
            unsafe { std::ptr::write_bytes(p.cast::<u8>(), 0, total_bytes); }
        }

        Ok(Self {
            base: p as *mut T,
            elems_per_slot,
            _marker: PhantomData,
        })
    }

    /// `frame_idx % N_SLOTS`. The slot the current DMA writes to.
    #[inline]
    pub const fn write_slot_index(frame_idx: u64) -> usize {
        (frame_idx % (Self::N_SLOTS as u64)) as usize
    }

    /// `(frame_idx - 2) % N_SLOTS`. The slot safe to read for the
    /// host serializer, given the current frame index. Returns
    /// `None` for `frame_idx < 2` (rotation history insufficient).
    #[inline]
    pub const fn read_slot_index(frame_idx: u64) -> Option<usize> {
        if frame_idx < 2 {
            None
        } else {
            Some(((frame_idx - 2) % (Self::N_SLOTS as u64)) as usize)
        }
    }

    /// Number of elements per slot. Fixed at construction time.
    #[inline]
    pub fn elems_per_slot(&self) -> usize {
        self.elems_per_slot
    }

    /// Bytes per slot. Fixed at construction time.
    #[inline]
    pub fn slot_bytes(&self) -> usize {
        self.elems_per_slot * std::mem::size_of::<T>()
    }

    /// Total pinned allocation in bytes (`N_SLOTS * slot_bytes`).
    #[inline]
    pub fn total_bytes(&self) -> usize {
        Self::N_SLOTS * self.slot_bytes()
    }

    /// Raw host pointer to the ring's base. For diagnostic / pinned-
    /// status verification only.
    #[inline]
    pub fn base_ptr(&self) -> *mut T {
        self.base
    }

    /// Pointer into the slot the current DMA should write to.
    /// Suitable as the `dst` argument to `cuMemcpyDtoHAsync_v2`.
    pub fn write_slot_dst_ptr(&self, frame_idx: u64) -> *mut c_void {
        let slot = Self::write_slot_index(frame_idx);
        unsafe { self.base.add(slot * self.elems_per_slot) as *mut c_void }
    }

    /// Slice into the slot safe-to-read at `current_frame_idx`.
    /// Returns `None` if the rotation history is < 2 frames.
    ///
    /// # Safety
    ///
    /// The ring does not own a synchronization primitive. The caller
    /// MUST have synchronized the telemetry stream (or recorded /
    /// queried a `cudaEvent_t` past the writing DMA) before reading
    /// the slice; otherwise the contents are torn / partial. Hence
    /// the `_unchecked` suffix.
    pub unsafe fn read_slot_unchecked(&self, current_frame_idx: u64) -> Option<&[T]> {
        let slot = Self::read_slot_index(current_frame_idx)?;
        if self.base.is_null() {
            return None;
        }
        Some(std::slice::from_raw_parts(
            self.base.add(slot * self.elems_per_slot),
            self.elems_per_slot,
        ))
    }
}

impl<T> Drop for PinnedTelemetryRing<T> {
    fn drop(&mut self) {
        if !self.base.is_null() && self.elems_per_slot > 0 {
            // Best-effort free; if cuMemFreeHost fails (extremely
            // rare; would mean a corrupted CUDA context), there's
            // nothing useful we can do from a Drop impl.
            unsafe {
                let _ = cuMemFreeHost(self.base as *mut c_void);
            }
        }
    }
}

// SAFETY: `T: Send` is enough — the ring's contents are POD or POD-
// like, and the ring itself owns the pinned allocation through Drop.
// The host pointer is dereferenced only via the slice accessors, both
// of which require an external synchronization barrier the caller
// provides. No interior mutability is exposed.
unsafe impl<T: Send> Send for PinnedTelemetryRing<T> {}

// ============================================================================
// Non-blocking telemetry stream
// ============================================================================

/// Create a CUDA stream with the `CU_STREAM_NON_BLOCKING` flag set.
///
/// Per operator mandate §2.1: any DMA launched on the legacy default
/// stream (Stream 0) implicitly synchronises with every other stream
/// on the device. The non-blocking flag breaks that anchor so the
/// telemetry DMA truly runs in parallel with the main MD integrator
/// stream and the Adjudicator stream.
///
/// Returns the raw `CUstream` handle on success; the caller is
/// responsible for `cuStreamDestroy` when the campaign ends.
pub fn create_non_blocking_telemetry_stream() -> Result<CUstream, i32> {
    let mut s: CUstream = std::ptr::null_mut();
    let rc = unsafe {
        cuStreamCreate(
            &mut s as *mut _,
            CUstream_flags_enum::CU_STREAM_NON_BLOCKING as u32,
        )
    };
    if !matches!(rc, CUresult::CUDA_SUCCESS) {
        return Err(rc as i32);
    }
    if s.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE as i32);
    }
    Ok(s)
}

/// Read back the flags currently set on a CUDA stream. Returned value
/// matches the `CU_STREAM_*` bitset (1 = NON_BLOCKING). Used by the
/// CSR Section L attestation that the telemetry stream is in fact
/// non-blocking.
pub fn stream_flags(stream: CUstream) -> Result<u32, i32> {
    let mut flags: u32 = 0;
    let rc = unsafe { cuStreamGetFlags(stream, &mut flags as *mut _) };
    if !matches!(rc, CUresult::CUDA_SUCCESS) {
        return Err(rc as i32);
    }
    Ok(flags)
}

/// Verify a host pointer is page-locked / pinned. Used by the CSR
/// Section L attestation that the telemetry ring is DMA-eligible.
///
/// Returns `Ok(true)` if `cuPointerGetAttribute` reports
/// `CU_MEMORYTYPE_HOST` for the pointer. `Ok(false)` if the pointer
/// is pageable (or not a registered host pointer at all). `Err` for
/// any underlying CUDA error.
pub fn is_pinned_host(host_ptr: *const c_void) -> Result<bool, i32> {
    let mut attr: u32 = 0;
    let rc = unsafe {
        cuPointerGetAttribute(
            &mut attr as *mut _ as *mut c_void,
            CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            host_ptr as CUdeviceptr,
        )
    };
    if !matches!(rc, CUresult::CUDA_SUCCESS) {
        // CUDA_ERROR_INVALID_VALUE here means "not a registered
        // CUDA pointer at all" — i.e. pageable. That is a
        // legitimate "not pinned" answer, not a hard error.
        if matches!(rc, CUresult::CUDA_ERROR_INVALID_VALUE) {
            return Ok(false);
        }
        return Err(rc as i32);
    }
    Ok(attr == (CUmemorytype_enum::CU_MEMORYTYPE_HOST as u32))
}

// ============================================================================
// Async tile copy (the actual DMA orchestration)
// ============================================================================

/// Schedule an asynchronous device → host copy of `n_elems` elements
/// of `T` from `src_dev_ptr` into the `frame_idx`-rotated write slot
/// of `ring`. Uses `cuMemcpyDtoHAsync_v2` on `telemetry_stream`.
///
/// The call is fire-and-forget: it does NOT synchronise the stream.
/// The ring's read accessor (`read_slot_unchecked`) makes the
/// `frame_idx % 3` rotation guarantee — the slot the host reads at
/// `i` was last written at `i - 2`, so by the time the host reads it
/// the DMA has had two full frames of slack to retire on the
/// non-blocking stream.
///
/// # Errors
///
/// Returns the raw cudaError code on failure (forwarded to the audit
/// spine). Common causes: `CUDA_ERROR_INVALID_VALUE` if `n_elems`
/// exceeds the ring's slot capacity, or `CUDA_ERROR_INVALID_HANDLE`
/// if the stream was destroyed.
pub fn schedule_async_tile_copy<T>(
    ring: &PinnedTelemetryRing<T>,
    src_dev_ptr: *const T,
    n_elems: usize,
    telemetry_stream: CUstream,
    frame_idx: u64,
) -> Result<(), i32> {
    if n_elems == 0 {
        return Ok(());
    }
    if n_elems > ring.elems_per_slot() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE as i32);
    }
    if src_dev_ptr.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE as i32);
    }
    let bytes = n_elems
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE as i32)?;

    let dst = ring.write_slot_dst_ptr(frame_idx);
    let rc =
        unsafe { cuMemcpyDtoHAsync_v2(dst, src_dev_ptr as CUdeviceptr, bytes, telemetry_stream) };
    if !matches!(rc, CUresult::CUDA_SUCCESS) {
        return Err(rc as i32);
    }
    Ok(())
}

// ============================================================================
// Host-side serializer — Pillar 5 reporting (operator addendum 2026-04-29)
//
// The asynchronous JSON serializer that the ghost-telemetry thread runs.
// Every tile in the `(frame_idx - 2) % 3` slot is converted into a
// `SiteManifest` stamp and written through a `BufWriter` — no GPU
// involvement, no allocation spikes that could jitter the MD orchestrator.
//
// The operator's mandate:
//   "All JSON formatting, string manipulation, and floating-point to
//    fixed-point conversions must occur on the CPU. The GPU must remain
//    'ignorant' of the serialization state. Use serde_json::to_writer_buffered
//    to ensure we don't hit memory allocation spikes on the host that
//    could jitter the MD orchestrator."
// ============================================================================

use std::io::{BufWriter, Write};

use crate::site_manifest::{
    CausalScalars, CentroidManifold, ClusterId, EntangledManifoldId, SiteId, SiteIdentity,
    SiteManifest,
};
use crate::so3_project::ContactShellTile;

/// One serialized telemetry record emitted per ContactShellTile read
/// from the ghost ring's safe-to-read slot. Each record stamps the
/// six-band normalized C_l power spectrum into the canonical
/// [`SiteManifest::contact_shell_geo_power_spectrum`] field along
/// with the cluster id and frame number.
///
/// Returns the number of bytes written by `serde_json::to_writer`
/// (BufWriter-buffered) for telemetry-rate accounting on the host.
///
/// # Surgical contract
///
/// * Constructs each [`SiteManifest`] via the existing
///   `from_lbvh_cluster_aabb` constructor when an AABB is available;
///   when not (typical: ghost-side reader has only the tile, not the
///   originating cluster's AABB), we fall back to a minimal
///   `SiteIdentity` + `EntangledManifoldId(frame)` shell so the
///   downstream JSON consumer still gets a valid record.
/// * Every Option field on `SiteManifest` other than the geometry-
///   plane C_l is left at `None` (operator I/O-bloat mitigation).
///   Adjudicator fields and KCC metrics are populated by separate
///   producers; this serializer only stamps the SO(3) ground truth.
pub fn serialize_slot_to_writer<W: Write>(
    ring: &PinnedTelemetryRing<ContactShellTile>,
    current_frame_idx: u64,
    writer: &mut BufWriter<W>,
) -> Result<usize, GhostSerializeError> {
    let slot = unsafe { ring.read_slot_unchecked(current_frame_idx) }
        .ok_or(GhostSerializeError::InsufficientHistory { current_frame_idx })?;

    let mut total_bytes = 0usize;
    // One JSON object per tile; line-delimited (NDJSON) for downstream
    // streaming readers (`jq -c .`, `simdjson::dom::parser::iterate`).
    for (cluster_id, tile) in slot.iter().enumerate() {
        let manifest = build_minimal_site_manifest_from_tile(cluster_id as u32, tile);
        let json = serde_json::to_string(&manifest).map_err(GhostSerializeError::Json)?;
        writer
            .write_all(json.as_bytes())
            .map_err(GhostSerializeError::Io)?;
        writer.write_all(b"\n").map_err(GhostSerializeError::Io)?;
        total_bytes += json.len() + 1;
    }
    Ok(total_bytes)
}

/// Build a minimal SiteManifest record from a ContactShellTile read
/// off the ghost ring. The reader doesn't carry per-cluster LBVH
/// AABBs, so this manifest:
///
/// * uses `cluster_id` derived from the tile's index in the ring slot
/// * stamps `tile.frame` into `SiteManifest::frame`
/// * stamps the geometry-plane C_l[0..6] into
///   `contact_shell_geo_power_spectrum`
/// * leaves every other extension field `None` (operator addendum
///   I/O-bloat mitigation: Cold-Hold tiles emit a near-empty record)
///
/// Pure-host code; no GPU involvement, no FFI calls.
fn build_minimal_site_manifest_from_tile(cluster_id: u32, tile: &ContactShellTile) -> SiteManifest {
    use crate::entangled_manifold::{
        CausalSignal, CausalSortKey, IdentityTieBreaker, SelectionPolicy, SortField,
        TieBreakerPolicy, ViewProvenance,
    };

    let provenance = ViewProvenance {
        signal: CausalSignal::SpikeAttributionCount,
        selection: SelectionPolicy::TopK { k: 1 },
        #[allow(deprecated)]
        tie_breaker: TieBreakerPolicy::CausalThenResid,
        frame: tile.frame as u64,
    };
    let sort_lineage = CausalSortKey {
        priorities: vec![SortField::SpikeAttributionCount],
        identity_tiebreaker: IdentityTieBreaker::ChainResidAtom,
    };

    // Stamp the published normalized C_l[0..6] from the geometry plane
    // (post-RECT-3.1.c L2-normalization + KL-EPS pad).
    let mut c_l = [0.0f32; 6];
    c_l.copy_from_slice(&tile.geo_power_spectrum[..6]);

    SiteManifest {
        identity: SiteIdentity {
            site_id: SiteId(cluster_id),
            cluster_id: ClusterId(cluster_id),
            provenance,
        },
        centroids: CentroidManifold::new(),
        causal_scalars: CausalScalars::new_m1(tile.spike_count as u64),
        frame: tile.frame as u64,
        source_manifold_id: EntangledManifoldId(tile.frame as u64),
        sort_lineage,
        contact_shell_geo_power_spectrum: Some(c_l),
        adjudicator_divergence: None,
        adjudicator_code: None,
        adjudicator_elapsed_ns: None,
        kcc_metrics: None,
        therm_dossier: None,
        ghost_phase_lattice: None,
        phase_manifold: None,
        therm_ccns_lifecycle: None,
        so3_manifold: None,
    }
}

/// Errors raised by [`serialize_slot_to_writer`].
#[derive(Debug)]
pub enum GhostSerializeError {
    /// Caller invoked the serializer before the ring had ≥ 2 frames
    /// of rotation history. The lagging slot doesn't exist yet.
    InsufficientHistory { current_frame_idx: u64 },
    /// Underlying `serde_json` failure (typically only on disk-full
    /// when the writer is a `BufWriter<File>`).
    Json(serde_json::Error),
    /// Underlying writer I/O failure.
    Io(std::io::Error),
}

impl std::fmt::Display for GhostSerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GhostSerializeError::InsufficientHistory { current_frame_idx } => write!(
                f,
                "ghost ring rotation history < 2 frames (current_frame_idx = {}); \
                 caller must wait until frame 2 before reading the lagging slot",
                current_frame_idx
            ),
            GhostSerializeError::Json(e) => write!(f, "serde_json error: {}", e),
            GhostSerializeError::Io(e) => write!(f, "writer io error: {}", e),
        }
    }
}

impl std::error::Error for GhostSerializeError {}

// ============================================================================
// F1 SWITCH — adjudication_code readback
// ============================================================================

/// Scan the ghost ring's safe-to-read slot for tiles where the
/// Adjudicator has set `adjudication_code == 1` (Burst / Construct
/// path — the F1 SWITCH fired).
///
/// Emits a structured `log::info!` line **with a microsecond-resolution
/// wall-clock timestamp** the exact moment each trigger is detected on
/// the host. Intended call site: immediately after `pipeline.launch()`
/// returns control to the host, so the DMA from frame `current_frame_idx - 2`
/// is guaranteed committed (the ring's 2-frame lag + the fact that
/// `pipeline.launch()` blocks until the graph launch is enqueued on the
/// stream, and the prior frame's DMA precedes the current launch in the
/// stream order).
///
/// # Returns
///
/// Number of tiles in the slot whose `adjudication_code` was 1.  A
/// non-zero return signals the operator's desired "Proof of Discovery":
/// the GPU decided a pocket is real and fired the ASC steering path.
///
/// # Safety
///
/// Caller must have either (a) synchronized the telemetry stream, or
/// (b) confirmed that `cuGraphLaunch` for the CURRENT frame has been
/// enqueued — which guarantees the DMA for frame `current_frame_idx - 2`
/// has completed (the ring protocol ensures 2-frame lag). Same contract
/// as `PinnedTelemetryRing::read_slot_unchecked`.
pub unsafe fn log_f1_switch_events(
    ring: &PinnedTelemetryRing<ContactShellTile>,
    current_frame_idx: u64,
    stream_id: usize,
) -> usize {
    let slot = match ring.read_slot_unchecked(current_frame_idx) {
        Some(s) => s,
        None => return 0,
    };
    let mut n_fired = 0usize;
    for tile in slot {
        if tile.adjudication_code == 1 {
            let ts_us = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros())
                .unwrap_or(0);
            log::info!(
                "[F1-SWITCH ts_us={} stream={} frame={} cluster={} \
                 adj=BURST C_l0={:.6e} spikes={}]",
                ts_us,
                stream_id,
                current_frame_idx.saturating_sub(2),
                tile.cluster_id,
                tile.geo_power_spectrum[0],
                tile.spike_count,
            );
            n_fired += 1;
        }
    }
    n_fired
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::so3_project::ContactShellTile;
    use cudarc::driver::sys::cuStreamSynchronize;

    // ---------- Pure host-side rotation tests (no CUDA needed) ----------

    #[test]
    fn write_slot_rotates_modulo_3() {
        assert_eq!(PinnedTelemetryRing::<u32>::write_slot_index(0), 0);
        assert_eq!(PinnedTelemetryRing::<u32>::write_slot_index(1), 1);
        assert_eq!(PinnedTelemetryRing::<u32>::write_slot_index(2), 2);
        assert_eq!(PinnedTelemetryRing::<u32>::write_slot_index(3), 0);
        assert_eq!(PinnedTelemetryRing::<u32>::write_slot_index(4), 1);
        assert_eq!(PinnedTelemetryRing::<u32>::write_slot_index(5), 2);
        // Frame 100 → slot 100%3 = 1.
        assert_eq!(PinnedTelemetryRing::<u32>::write_slot_index(100), 1);
    }

    #[test]
    fn build_minimal_site_manifest_from_tile_stamps_geo_cl_only() {
        // Pure-host: no GPU needed. Verifies the ghost serializer
        // converts a synthetic tile into a minimal SiteManifest with
        // exactly the geometry-plane C_l populated; every other
        // optional field stays None per the I/O-bloat mitigation.
        let mut tile = ContactShellTile::zero();
        tile.frame = 99;
        tile.spike_count = 16;
        // Synthetic L2-normalized geometry C_l (sums to 1.0).
        tile.geo_power_spectrum = [0.40, 0.30, 0.15, 0.08, 0.04, 0.03, 0.0, 0.0];

        let m = super::build_minimal_site_manifest_from_tile(7, &tile);
        assert_eq!(m.frame, 99);
        assert_eq!(m.identity.cluster_id.0, 7);
        assert_eq!(m.identity.site_id.0, 7);
        assert_eq!(m.causal_scalars.spike_attribution_count, 16);
        let cl = m.contact_shell_geo_power_spectrum.expect("Some(C_l)");
        assert_eq!(cl, [0.40, 0.30, 0.15, 0.08, 0.04, 0.03]);
        // Every other extension field must stay None — Pillar-5 reporter
        // never speaks for the Adjudicator nor the KCC pipeline.
        assert!(m.adjudicator_divergence.is_none());
        assert!(m.adjudicator_code.is_none());
        assert!(m.adjudicator_elapsed_ns.is_none());
        assert!(m.kcc_metrics.is_none());
        assert!(m.therm_dossier.is_none());

        // Serialize → assert the JSON contains exactly the expected
        // shape (no Adjudicator/KCC/Therm leakage).
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains("\"contact_shell_geo_power_spectrum\""));
        assert!(s.contains("\"frame\":99"));
        for forbidden in [
            "adjudicator_divergence",
            "adjudicator_code",
            "adjudicator_elapsed_ns",
            "kcc_metrics",
            "therm_dossier",
        ] {
            assert!(
                !s.contains(forbidden),
                "Pillar-5 ghost reader leaked extension field {}: {}",
                forbidden,
                s
            );
        }
    }

    #[test]
    fn read_slot_lags_by_two_frames() {
        // Frames 0, 1: rotation history insufficient → None.
        assert_eq!(PinnedTelemetryRing::<u32>::read_slot_index(0), None);
        assert_eq!(PinnedTelemetryRing::<u32>::read_slot_index(1), None);
        // Frame 2: read slot 0 (frame 0's write).
        assert_eq!(PinnedTelemetryRing::<u32>::read_slot_index(2), Some(0));
        // Frame 3: read slot 1 (frame 1's write).
        assert_eq!(PinnedTelemetryRing::<u32>::read_slot_index(3), Some(1));
        // Frame 4: read slot 2 (frame 2's write).
        assert_eq!(PinnedTelemetryRing::<u32>::read_slot_index(4), Some(2));
        // Frame 5: read slot 0 (frame 3's write — slot 0 was overwritten).
        assert_eq!(PinnedTelemetryRing::<u32>::read_slot_index(5), Some(0));
    }

    // ---------- GPU-side: pinned-status + stream-flag attestation ----------

    #[test]
    fn ring_allocation_is_pinned_and_nonzero_size() {
        use cudarc::driver::CudaContext;
        let _ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[ghost-pinned] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };

        // 16 ContactShellTiles per slot × 3 slots × 1280 B = 60 KB.
        let ring: PinnedTelemetryRing<ContactShellTile> =
            PinnedTelemetryRing::new(16).expect("alloc pinned ring");
        assert_eq!(ring.elems_per_slot(), 16);
        assert_eq!(ring.slot_bytes(), 16 * 1280);
        assert_eq!(ring.total_bytes(), 3 * 16 * 1280);
        assert!(!ring.base_ptr().is_null());

        let pinned =
            is_pinned_host(ring.base_ptr() as *const c_void).expect("cuPointerGetAttribute");
        assert!(
            pinned,
            "ring base pointer must be pinned (CU_MEMORYTYPE_HOST)"
        );

        // Each slot's pointer is also in pinned memory.
        for f in 0..3u64 {
            let dst = ring.write_slot_dst_ptr(f);
            let pinned = is_pinned_host(dst).expect("cuPointerGetAttribute");
            assert!(pinned, "slot {} pointer not pinned", f);
        }
    }

    #[test]
    fn telemetry_stream_is_non_blocking() {
        use cudarc::driver::CudaContext;
        let _ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[ghost-stream] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream =
            create_non_blocking_telemetry_stream().expect("create non-blocking telemetry stream");
        let flags = stream_flags(stream).expect("cuStreamGetFlags");
        // CU_STREAM_NON_BLOCKING == 1; the default-stream-anchored
        // bit is 0 (CU_STREAM_DEFAULT).
        assert_eq!(
            flags & 1,
            1,
            "telemetry stream missing CU_STREAM_NON_BLOCKING (got flags=0x{:x})",
            flags
        );
        unsafe {
            let _ = cudarc::driver::sys::cuStreamDestroy_v2(stream);
        }
    }

    // ---------- End-to-end: schedule_async_tile_copy ----------

    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn end_to_end_async_copy_round_trips_payload() {
        use crate::rich_spike::RichSpike;
        use crate::so3_project::{
            ContactShellTile, SiteManifestFfi, So3ProjectInput, So3ProjectTransform,
        };
        use crate::transform::{AuditOutcome, AuditedTransform};
        use crate::vram_pool::VramPool;
        use cudarc::driver::{CudaContext, DevicePtr};

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[ghost-e2e] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        // Main MD stream (the captured-graph stream in production).
        let md_stream = ctx.new_stream().expect("md stream");
        let md_raw = md_stream.cu_stream() as usize;

        // Init K_LM, get device pointer.
        unsafe {
            let _ = crate::sh_basis::ffi::prism_sh_basis_init(md_raw as *mut c_void);
        }
        md_stream.synchronize().expect("post-sh-init sync");
        let k_lm_dev = crate::sh_basis::k_lm_device_ptr().expect("k_lm");

        // Telemetry stream (the GHOST stream).
        let telemetry_stream =
            create_non_blocking_telemetry_stream().expect("create telemetry stream");

        // F2 pool for the device-side ContactShellTile array.
        let pool = match VramPool::new(0) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[ghost-e2e] VramPool::new failed: {} — skipping", e);
                unsafe {
                    let _ = cudarc::driver::sys::cuStreamDestroy_v2(telemetry_stream);
                }
                return;
            }
        };

        const N_CLUSTERS: u32 = 1;
        let alloc_bytes = SiteManifestFfi::alloc_bytes(N_CLUSTERS);
        let tiles_ptr_u = pool.alloc_async(alloc_bytes, md_raw).expect("F2 alloc");
        md_stream.synchronize().expect("post-alloc sync");

        let mut manifest = SiteManifestFfi {
            total_sites: N_CLUSTERS,
            _pad0: 0,
            tiles_dev_ptr: tiles_ptr_u as *mut ContactShellTile,
            vram_high_water_mark: alloc_bytes,
            adjudication_trigger_ptr: std::ptr::null_mut(),
        };

        // Synthesize a tiny single-cluster spike buffer.
        let spikes: Vec<RichSpike> = (0..16u32)
            .map(|i| {
                let mut s = RichSpike::zero();
                let theta = 0.3 + (i as f32) * 0.2;
                let phi = 0.4 + (i as f32) * 0.3;
                s.x = 4.0 * theta.sin() * phi.cos();
                s.y = 4.0 * theta.sin() * phi.sin();
                s.z = 4.0 * theta.cos();
                s.cluster_id = 0;
                s
            })
            .collect();
        let offsets: Vec<u32> = vec![0u32, spikes.len() as u32];

        let spike_bytes = spikes.len() * std::mem::size_of::<RichSpike>();
        let mut d_spikes_b = md_stream.alloc_zeros::<u8>(spike_bytes).expect("alloc");
        let spikes_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(spikes.as_ptr() as *const u8, spike_bytes).to_vec()
        };
        md_stream
            .memcpy_htod(&spikes_bytes, &mut d_spikes_b)
            .expect("htod");
        let mut d_offsets = md_stream
            .alloc_zeros::<u32>(offsets.len())
            .expect("alloc o");
        md_stream
            .memcpy_htod(&offsets, &mut d_offsets)
            .expect("htod o");
        let (sp_dev, _g1) = d_spikes_b.device_ptr(&md_stream);
        let (off_dev, _g2) = d_offsets.device_ptr(&md_stream);

        // Run the SO(3) projection on the MD stream — fills the F2-pool
        // tile buffer.
        let outcome = So3ProjectTransform::new().apply(So3ProjectInput {
            pool_handle: pool.raw_handle(),
            stream_handle: md_raw,
            d_spikes: sp_dev as *const RichSpike,
            d_cluster_offsets: off_dev as *const u32,
            n_clusters: N_CLUSTERS,
            d_k_lm: k_lm_dev,
            frame_id: 0,
            source_stream_id: 0,
            manifest: &mut manifest,
        });
        match outcome {
            AuditOutcome::Accepted { .. } => (),
            AuditOutcome::Quarantined { violations, .. }
            | AuditOutcome::Aborted { violations, .. } => {
                pool.free_async(tiles_ptr_u, md_raw).ok();
                unsafe {
                    let _ = cudarc::driver::sys::cuStreamDestroy_v2(telemetry_stream);
                }
                panic!("SO(3) failed: {:?}", violations);
            }
        }
        md_stream.synchronize().expect("md sync");

        // CLA-2: ghost ring + async DMA on the telemetry stream.
        let ring: PinnedTelemetryRing<ContactShellTile> =
            PinnedTelemetryRing::new(N_CLUSTERS as usize).expect("ring alloc");

        // Frame 0: write slot 0. Frames 1, 2: write slots 1, 2.
        // After frame 2's write completes we can read slot 0 = frame 0.
        for frame in 0..3u64 {
            schedule_async_tile_copy(
                &ring,
                manifest.tiles_dev_ptr as *const ContactShellTile,
                N_CLUSTERS as usize,
                telemetry_stream,
                frame,
            )
            .expect("async d2h");
        }
        // Sync the GHOST stream — does NOT touch the MD stream.
        let rc = unsafe { cuStreamSynchronize(telemetry_stream) };
        assert!(matches!(rc, CUresult::CUDA_SUCCESS), "ghost stream sync");

        // After 3 frames written, frame 2 is the "current frame";
        // read slot index = (2-2)%3 = 0, which holds frame 0's tile.
        let slot = unsafe { ring.read_slot_unchecked(2) }
            .expect("rotation should yield Some at frame >= 2");
        assert_eq!(slot.len(), N_CLUSTERS as usize);
        // The kernel stamped frame_id = 0 into the tile header.
        assert_eq!(slot[0].frame, 0);
        assert_eq!(slot[0].spike_count, spikes.len() as u32);

        // Cross-frame guarantee: frames 1 and 2 went to slots 1 and 2.
        // Their tile.frame must echo (1 was launched with frame_id=0
        // because we re-used the same launch — but the data is consistent).
        // Specifically: every slot must hold a sane tile.
        for s in 0..3 {
            let tile =
                unsafe { std::ptr::read_unaligned(ring.base_ptr().add(s * ring.elems_per_slot())) };
            assert_eq!(
                tile.spike_count,
                spikes.len() as u32,
                "slot {} corrupted: spike_count = {}",
                s,
                tile.spike_count
            );
        }

        eprintln!(
            "[ghost-e2e] CLA-2 verified: pinned ring × 3, non-blocking \
                  stream, 3-frame async DMA, slot[0] holds frame 0's tile \
                  (frame={}, spike_count={})",
            slot[0].frame, slot[0].spike_count
        );

        // Cleanup.
        pool.free_async(tiles_ptr_u, md_raw).ok();
        md_stream.synchronize().ok();
        unsafe {
            let _ = cudarc::driver::sys::cuStreamDestroy_v2(telemetry_stream);
        }
    }
}
