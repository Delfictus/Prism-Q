//! ZSTR — Zero-Stall Telemetry Ring
//!
//! Implements the "Compute-Stream-Consume" exfiltration architecture for
//! V2 coordinate snapshots.  Three asynchronous planes:
//!
//! 1. **Device Plane** (CUDA graph) — physics + adjudication + position
//!    staging via `zstr_pos_stage_f4_kernel` + fence via
//!    `zstr_signal_completion_kernel`.
//!
//! 2. **Exfiltration Plane** (ZSTR consumer thread) — spins on pinned
//!    `completion_fence`; on READY, writes the frame to NVMe via
//!    `O_DIRECT` (bypasses page cache), resets the fence, and advances
//!    the ring index.
//!
//! 3. **Control Plane** (Rust orchestrator) — allocates the ring before
//!    the CUDA graph capture, passes device-side pointers into the graph
//!    nodes, spawns the consumer thread, and joins it at teardown.
//!
//! # Triple-buffer ring geometry
//!
//! ```text
//! slot = frame_idx % 3
//!
//! slot N   : GPU writing (current cudaMemcpyAsync target)
//! slot N-1 : in-flight / verification buffer
//! slot N-2 : host consumer locked for O_DIRECT write
//! ```
//!
//! # Gate G21 — Alignment verification
//!
//! `ZstrRing::allocate` asserts `pinned_ptr % 4096 == 0` and sets
//! `alignment_ok`.  The consumer thread panics if the gate is not passed
//! before it attempts any `O_DIRECT` write.
//!
//! # Gate G22 — Atomic v4 (CUDA kernel side)
//!
//! See `asc_steering.cu` for `atom.global.add.v4.f32` PTX; run:
//! `nvcc -arch=sm_120 --ptx asc_steering.cu | grep "atom.global.add.v4"`
//! to close G22.

#![cfg(feature = "gpu")]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

// ─── ZstrFrameHeader ────────────────────────────────────────────────────────

/// Header for one ZSTR ring slot (pinned, 4096-byte aligned).
///
/// T11 (Action-Recovery): the slot now carries forces in addition to
/// positions; the header gains `n_atoms`, `gear_id`, and `force_norm`
/// so the offline replay can reconstruct (a) the system size, (b) the
/// G26 chronometric gear active during the frame (set by Wave B; default 0
/// in Wave A), and (c) the L2 norm of the integrator's total force after
/// ASC steering — a single scalar witness that the steering is bounded
/// and finite.
///
/// Immediately followed in the same allocation by:
///   [positions: n_atoms * 3 * f32]
///   [forces:    n_atoms * 3 * f32]
///   [padding to next 4096-byte boundary]
///
/// The full slot size is computed and padded to 4096 bytes so that
/// `O_DIRECT` writes are always sector-aligned.
#[repr(C, align(4096))]
pub struct ZstrFrameHeader {
    /// Monotonically increasing epoch counter (0-based).
    pub frame_idx: u64, // 0..8
    /// Integration timestep in picoseconds at the time of capture.
    pub dt: f32, // 8..12
    /// adjudication_code written by the InterferometricAdjudicator
    /// (0 = Prune, 1 = Construct, 2 = Violation).
    pub adjudication_code: u32, // 12..16
    /// 0 = Dirty/Writing (GPU owns).  1 = Ready (host may flush).
    /// Written by `zstr_signal_completion_kernel` after __threadfence_system().
    pub completion_fence: u32, // 16..20
    /// Atom count for this slot (replay payload size = n_atoms * 3 * 4
    /// bytes for positions and again for forces). Stamped at capture
    /// time so the consumer doesn't need a side-channel size.
    pub n_atoms: u32, // 20..24
    /// G26 chronometric gear in effect when this frame retired.
    /// 0 = 0.5 fs (high-res), 1 = 2.0 fs (default), 2 = 4.0 fs (HMR sprint),
    /// 3 = abort-trap. Wave A leaves this at 0 (gear logic is Wave B);
    /// the field is reserved here so no header layout change is needed
    /// when the SWITCH lands.
    pub gear_id: u32, // 24..28
    /// L2 norm ‖F‖₂ = sqrt(Σ_i Σ_d F_id²) of the post-ASC integrator
    /// force buffer at this slot. NaN ⇒ G29 trap (steering produced
    /// non-physical forces; campaign aborts on detection).
    pub force_norm: f32, // 28..32
    /// **M1.2.18.5 — Total External Work (kcal/mol)** integrated over
    /// this chunk window.  Snapshotted from `*adj.d_external_work`
    /// (FFI offset 128 → F2-pool W_ext buffer) at frame retirement.
    /// `nan` is reserved (write zero when the W_ext pointer is null
    /// pre-M1.2.18.5 wire-up).  Off-line replay subtracts this from
    /// `potential_energy` deltas to derive the conservative-only
    /// component of ΔV (First Law audit).
    pub external_work: f64, // 32..40
    /// **M1.2.18.5 — System Potential Energy V_t (kcal/mol)** as written
    /// by the captured Hamiltonian-Auditor reduce node into
    /// `adj.d_potential_energy` (FFI offset 112).  Snapshotted at frame
    /// retirement so off-line replay can rebuild the V trace without
    /// re-reducing per-atom components.  Zero pre-Hamiltonian wire-up.
    pub potential_energy: f64, // 40..48
    /// **WP4** — id of the telemetry stream that produced this record.
    /// Carved out of the padding tail (offsets 0..48 are frozen and
    /// consumed by CUDA kernels + the offline parser). Stamped by
    /// `prism_zstr_device_slot_update_kernel`. Lets a multi-stream
    /// campaign attribute a record without relying on the file name.
    pub stream_id: u32, // 48..52
    /// **WP4** — ZSTR header schema version. 1 = the WP4 stamped-header
    /// contract (frame_idx / dt / n_atoms / gear_id / adjudication_code
    /// all written on device). 0 = pre-WP4 unstamped output, which is
    /// unparseable past the first 4096 bytes.
    pub schema_version: u32, // 52..56
    /// Pads the header to a full 4096-byte sector. align(4096) on the
    /// struct already enforces tail padding, but making it explicit
    /// here pins the slot layout so the offline replay format does not
    /// silently drift if the struct is later extended.
    /// M1.2.18.5: 1016 → 1012 (lost 16 B = 4 × u32 to external_work +
    /// potential_energy f64 fields above).
    /// WP4: 1012 → 1010 (lost 8 B = 2 × u32 to stream_id +
    /// schema_version above). No existing field moved.
    pub _padding: [u32; 1010], // 56..4096
}

// ZstrFrameHeader meaningful payload (post-M1.2.18.5): 48 bytes.
// size_of::<ZstrFrameHeader>() == 4096 (explicit + align(4096)).
// Slot layout post-T11:
//   [header: 4096 B] [positions: n_atoms*3*4 B] [forces: n_atoms*3*4 B] [pad→4096]

// ─── ZstrRing ────────────────────────────────────────────────────────────────

/// Triple-buffered pinned-host ring for ZSTR frame exfiltration.
///
/// Owns a single `cuMemHostAlloc` allocation with:
/// - a raw allocation base pointer used only for `cuMemFreeHost`
/// - a 4096-aligned usable base pointer used for ring payload access.
///
/// The usable region covers exactly `frame_size * N_SLOTS` bytes.
pub struct ZstrRing {
    /// Raw base returned by `cuMemHostAlloc` (must be freed as-is).
    pub raw_pinned_ptr: *mut u8,
    /// 4096-aligned usable base for all ring operations.
    pub pinned_ptr: *mut u8,
    /// Per-slot byte size (4096-byte padded).
    pub frame_size: usize,
    /// Atom count (determines position payload size).
    pub n_atoms: usize,
    /// G21 gate: true iff `pinned_ptr % 4096 == 0`.
    pub alignment_ok: bool,
    /// **WP4** — base of the per-stream control sector, a 4096-byte tail
    /// region of the same pinned + DEVICEMAP allocation. Holds the active
    /// slot cell and the backpressure/conservation status block.
    ///
    /// Pinned rather than device memory on purpose: the ZSTR consumer runs
    /// on a plain `std::thread` with no CUDA context current, so it cannot
    /// call `cuMemcpyDtoH`. Device-mapped pinned memory lets the kernels
    /// write it and the consumer read it with a volatile load.
    pub control_ptr: *mut u8,
}

unsafe impl Send for ZstrRing {}
unsafe impl Sync for ZstrRing {}

impl ZstrRing {
    /// Number of ring slots (5-slot resilience ring — Amendment 2.2).
    /// 5 slots × ~61 KB/frame = ~305 KB pinned headroom vs. P99 NVMe jitter.
    /// Contention policy: frame dropped (not queued) when slot still dirty.
    pub const N_SLOTS: usize = 5;

    /// **WP4** — size of the per-stream control sector appended after the
    /// ring slots. One 4096-byte sector keeps the whole allocation
    /// O_DIRECT-friendly and every field naturally 8-byte aligned.
    pub const CONTROL_BYTES: usize = 4096;

    /// **WP4** — offset of the `uint32_t` active-slot cell inside the
    /// control sector.
    pub const CONTROL_OFF_ACTIVE_SLOT: usize = 0;

    /// **WP4** — offset of the backpressure/conservation status block
    /// inside the control sector. 64 keeps it on its own cache line, away
    /// from the slot cell that the staging kernels poll every epoch.
    pub const CONTROL_OFF_STATUS: usize = 64;

    /// Allocate the triple-buffer in pinned host memory.
    ///
    /// Frame layout per slot (T11 — forces co-resident with positions):
    /// ```text
    /// [ZstrFrameHeader: 4096 B] [positions: n_atoms*3*4 B]
    ///   [forces: n_atoms*3*4 B] [pad→next 4096 multiple]
    /// ```
    pub fn allocate(n_atoms: usize) -> Result<Self, String> {
        // ZstrFrameHeader is align(4096) → size_of = 4096 (padded by Rust).
        // Meaningful header data occupies only the first 32 bytes.
        let header_bytes = std::mem::size_of::<ZstrFrameHeader>(); // == 4096
                                                                   // Operator Amendment 3.11 — ZSTR Payload Alignment Rectification.
                                                                   //
                                                                   // `zstr_force_stage_f4_kernel` issues `STG.E.128` (float4 store) at
                                                                   // base `slot_base + force_offset_in_slot`.  Blackwell sm_120
                                                                   // requires the destination address to be 16-byte aligned or the
                                                                   // hardware traps with MISALIGNED_ADDRESS (kernel-printf triage
                                                                   // 2026-05-02 isolated the trap to this kernel).
                                                                   //
                                                                   // For any `n_atoms` where `n_atoms mod 4 != 0`, the raw pos
                                                                   // payload size `n_atoms * 12` is not a multiple of 16, leaving
                                                                   // `force_offset = 4096 + 12·n_atoms` at an 8-aligned (not 16-
                                                                   // aligned) offset within the slot.  Padding the pos payload up
                                                                   // to the next 16-byte boundary fixes the offset for the force
                                                                   // payload — the slot is still 4096-pad'd at the end so the
                                                                   // O_DIRECT total size is preserved.
        let pos_bytes_raw = n_atoms * 3 * 4; // n_atoms × xyz × f32
        let pos_bytes_aligned = (pos_bytes_raw + 15) & !15;
        let force_bytes = n_atoms * 3 * 4; // T11: forces appended
        let raw_frame = header_bytes + pos_bytes_aligned + force_bytes;
        // Round up to next 4096-byte multiple for O_DIRECT sector alignment.
        let frame_size = (raw_frame + 4095) & !4095;
        // WP4 — the usable region is N_SLOTS frames followed by one
        // 4096-byte per-stream control sector (active slot + backpressure /
        // conservation counters). Carving it out of the same pinned +
        // DEVICEMAP allocation keeps ZstrRing a single-owner resource and
        // leaves `Drop for ZstrRing` unchanged.
        let total_bytes = frame_size * Self::N_SLOTS + Self::CONTROL_BYTES;
        let alloc_bytes = total_bytes + 4095;

        // M1.2.19.D — Mapped-memory standard (operator Amendment 3.13 §1.2).
        // Use cuMemHostAlloc with PORTABLE + DEVICEMAP so the page-locked
        // host buffer carries an explicit device-side virtual-address alias
        // (vs. the implicit-mapped behavior of cuMemAllocHost_v2).  On
        // Blackwell sm_120 + PCIe Gen5 the driver collapses them to the
        // same VA window, but the explicit flag makes the contract part
        // of the source — required for multi-node B200 portability where
        // the implicit behavior is not guaranteed.
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let flags = cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE
            | cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP;
        let rc = unsafe { cudarc::driver::sys::cuMemHostAlloc(&mut ptr, alloc_bytes, flags) };

        if !matches!(rc, cudarc::driver::sys::CUresult::CUDA_SUCCESS) {
            return Err(format!(
                "cuMemHostAlloc(PORTABLE|DEVICEMAP) failed: {:?} \
                 (requested {} bytes usable / {} bytes allocated for {} atoms × {} slots)",
                rc,
                total_bytes,
                alloc_bytes,
                n_atoms,
                Self::N_SLOTS
            ));
        }

        let raw_pinned_ptr = ptr as *mut u8;
        let raw_addr = raw_pinned_ptr as usize;
        let aligned_addr = (raw_addr + 4095) & !4095usize;
        let alignment_offset = aligned_addr.saturating_sub(raw_addr);
        let pinned_ptr = aligned_addr as *mut u8;
        let alignment_ok = (pinned_ptr as usize) % 4096 == 0;

        // Zero-init: clears all completion_fence fields to 0 (GPU-owned).
        unsafe {
            std::ptr::write_bytes(raw_pinned_ptr, 0u8, alloc_bytes);
        }

        // WP4 — control sector sits immediately after the last slot.
        let control_ptr = unsafe { pinned_ptr.add(frame_size * Self::N_SLOTS) };
        // No slot is claimed until the in-graph slot-authority kernel runs.
        // Zeroed memory would read as "slot 0 claimed", which would let the
        // staging kernels write slot 0 before any header was stamped.
        unsafe {
            std::ptr::write_volatile(
                control_ptr.add(Self::CONTROL_OFF_ACTIVE_SLOT) as *mut u32,
                ZSTR_SLOT_NONE,
            );
        }

        log::info!(
            "[ZSTR G21] raw_ptr={:p} aligned_ptr={:p} align_offset={} \
             frame_size={} usable_bytes={} alloc_bytes={} n_atoms={} \
             alignment_pass={}",
            raw_pinned_ptr,
            pinned_ptr,
            alignment_offset,
            frame_size,
            total_bytes,
            alloc_bytes,
            n_atoms,
            alignment_ok
        );
        if alignment_ok {
            log::info!(
                "[ZSTR G21 PASS] aligned_ptr {:p} 4096-aligned ✓",
                pinned_ptr
            );
        } else {
            log::error!(
                "[ZSTR G21 FAIL] aligned_ptr {:p} not 4096-aligned \
                 (raw_ptr={:p}, offset={}) — O_DIRECT writes will fault.",
                pinned_ptr,
                raw_pinned_ptr,
                alignment_offset
            );
        }

        Ok(Self {
            raw_pinned_ptr,
            pinned_ptr,
            frame_size,
            n_atoms,
            alignment_ok,
            control_ptr,
        })
    }

    // ─── WP4 per-stream control sector accessors ────────────────────────────

    /// Device/host pointer to the per-stream `uint32_t` active-slot cell.
    /// Unified addressing (the allocation is PORTABLE|DEVICEMAP) makes the
    /// host pointer valid as a device pointer, exactly as
    /// `positions_cu_ptr` already assumes.
    pub fn active_slot_ptr(&self) -> *mut u32 {
        unsafe { self.control_ptr.add(Self::CONTROL_OFF_ACTIVE_SLOT) as *mut u32 }
    }

    /// Device/host pointer to the per-stream backpressure/conservation
    /// status block (`ZSTR_BP_OFF_*` layout).
    pub fn backpressure_status_ptr(&self) -> *mut u8 {
        unsafe { self.control_ptr.add(Self::CONTROL_OFF_STATUS) }
    }

    fn status_u32(&self, off: usize) -> u32 {
        unsafe { std::ptr::read_volatile(self.backpressure_status_ptr().add(off) as *const u32) }
    }

    fn status_u64(&self, off: usize) -> u64 {
        unsafe { std::ptr::read_volatile(self.backpressure_status_ptr().add(off) as *const u64) }
    }

    /// Current device backpressure error code (`ZSTR_BP_ERR_*`).
    pub fn backpressure_error(&self) -> u32 {
        self.status_u32(ZSTR_BP_OFF_ERROR)
    }

    /// Slot that tripped the bounded wait (valid iff `backpressure_error != 0`).
    pub fn backpressure_error_slot(&self) -> u32 {
        self.status_u32(ZSTR_BP_OFF_SLOT)
    }

    /// Number of slot-authority kernel invocations — i.e. the number of
    /// coordinate epochs the device actually executed for this stream.
    /// This is the `expected` term of the ZSTR conservation identity.
    pub fn device_epochs(&self) -> u64 {
        self.status_u64(ZSTR_BP_OFF_EPOCHS)
    }

    /// Number of epochs that successfully claimed a slot, staged a payload
    /// and raised the completion fence. The `produced` term.
    pub fn device_produced(&self) -> u64 {
        self.status_u64(ZSTR_BP_OFF_PRODUCED)
    }

    /// Number of epochs whose claim was refused by the bounded wait.
    /// `device_epochs() == device_produced() + device_backpressure_events()`
    /// on a well-formed run.
    pub fn device_backpressure_events(&self) -> u64 {
        self.status_u64(ZSTR_BP_OFF_EVENTS)
    }

    /// Current value of the per-stream active-slot cell (`ZSTR_SLOT_NONE`
    /// when no epoch holds a claim).
    pub fn active_slot(&self) -> u32 {
        unsafe { std::ptr::read_volatile(self.active_slot_ptr() as *const u32) }
    }

    /// Pointer to the `ZstrFrameHeader` for ring slot `slot` (0..N_SLOTS).
    ///
    /// # Safety
    /// Caller must ensure no concurrent GPU write to the same slot.
    pub unsafe fn header_ptr(&self, slot: usize) -> *mut ZstrFrameHeader {
        debug_assert!(slot < Self::N_SLOTS);
        self.pinned_ptr.add(slot * self.frame_size) as *mut ZstrFrameHeader
    }

    /// Raw byte pointer to the positions payload inside slot `slot`.
    /// This is what `zstr_pos_stage_f4_kernel` writes into.
    ///
    /// Offset = slot * frame_size + sizeof(ZstrFrameHeader).
    pub fn positions_host_ptr(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < Self::N_SLOTS);
        let header_off = std::mem::size_of::<ZstrFrameHeader>();
        unsafe { self.pinned_ptr.add(slot * self.frame_size + header_off) }
    }

    /// Device-side CUdeviceptr for the positions payload of slot `slot`.
    /// Pass this to `cuMemcpyDtoHAsync_v2` as the destination.
    /// (Pinned memory is directly addressable from the GPU.)
    pub fn positions_cu_ptr(&self, slot: usize) -> u64 {
        self.positions_host_ptr(slot) as u64
    }

    /// Byte offset from a slot's base to the forces payload.
    /// Header (4096) + positions payload aligned up to 16-byte boundary.
    ///
    /// Operator Amendment 3.11 — pos payload `n_atoms*12` is rounded up
    /// to the next 16-byte boundary so this offset (and therefore the
    /// `slot_base + force_offset` address consumed by
    /// `zstr_force_stage_f4_kernel`) is 16-aligned for `STG.E.128`.
    /// Without padding, any `n_atoms mod 4 != 0` produced an 8-aligned
    /// offset → MISALIGNED_ADDRESS hardware trap on the first launch.
    pub fn forces_offset_in_slot(&self) -> usize {
        let header_bytes = std::mem::size_of::<ZstrFrameHeader>();
        let pos_bytes_raw = self.n_atoms * 3 * 4;
        let pos_bytes_aligned = (pos_bytes_raw + 15) & !15;
        header_bytes + pos_bytes_aligned
    }

    /// Raw byte pointer to the forces payload inside slot `slot` (T11).
    /// This is what `zstr_force_stage_f4_kernel` writes into.
    pub fn forces_host_ptr(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < Self::N_SLOTS);
        unsafe {
            self.pinned_ptr
                .add(slot * self.frame_size + self.forces_offset_in_slot())
        }
    }

    /// Device-side CUdeviceptr for the forces payload of slot `slot` (T11).
    pub fn forces_cu_ptr(&self, slot: usize) -> u64 {
        self.forces_host_ptr(slot) as u64
    }

    /// Byte offset of `force_norm` within `ZstrFrameHeader`. Stable: 28.
    pub const fn force_norm_offset_in_slot() -> usize {
        offset_of_force_norm()
    }

    /// Device-side CUdeviceptr for the `completion_fence` field of slot `slot`.
    /// Pass to `zstr_signal_completion_kernel` as its argument.
    pub fn fence_cu_ptr(&self, slot: usize) -> u64 {
        let hdr = unsafe { self.header_ptr(slot) };
        // completion_fence offset within ZstrFrameHeader
        let field_off = offset_of_completion_fence();
        (hdr as u64) + field_off as u64
    }
}

impl Drop for ZstrRing {
    fn drop(&mut self) {
        if !self.raw_pinned_ptr.is_null() {
            unsafe {
                let _ = cudarc::driver::sys::cuMemFreeHost(self.raw_pinned_ptr as *mut c_void);
            }
            self.raw_pinned_ptr = std::ptr::null_mut();
            self.pinned_ptr = std::ptr::null_mut();
            // WP4 — the control sector is a tail region of the same
            // allocation, not a separate one; only the alias is cleared.
            self.control_ptr = std::ptr::null_mut();
        }
    }
}

/// Byte offset of `completion_fence` within the raw header bytes.
/// frame_idx(8) + dt(4) + adjudication_code(4) = 16.
const fn offset_of_completion_fence() -> usize {
    16
}

/// Byte offset of `force_norm` within the raw header bytes.
/// frame_idx(8) + dt(4) + adjudication_code(4) + completion_fence(4)
/// + n_atoms(4) + gear_id(4) = 28.
const fn offset_of_force_norm() -> usize {
    28
}

/// **M1.2.18.5** Byte offset of `external_work` within the raw header bytes.
/// `force_norm` ends at 32; `external_work` (f64) lives at 32..40.
pub const fn offset_of_external_work() -> usize {
    32
}

/// **M1.2.18.5** Byte offset of `potential_energy` within the raw header bytes.
/// `external_work` ends at 40; `potential_energy` (f64) lives at 40..48.
pub const fn offset_of_potential_energy() -> usize {
    40
}

// Compile-time pin: the header layout MUST match the offsets the C-side
// kernels and the offline replay parser depend on.  Drift here ⇒ silently
// corrupt force_norm / gear_id reads in the consumer thread.
const _: () = {
    use std::mem::size_of;
    assert!(size_of::<ZstrFrameHeader>() == 4096);
    // M1.2.18.5 — header packed payload is 48 bytes:
    //   frame_idx       u64       8 B
    //   dt              f32       4 B
    //   adjudication    u32       4 B
    //   completion      u32       4 B
    //   n_atoms         u32       4 B
    //   gear_id         u32       4 B
    //   force_norm      f32       4 B
    //   external_work   f64       8 B
    //   potential_energy f64      8 B
    //   stream_id       u32       4 B   (WP4)
    //   schema_version  u32       4 B   (WP4)
    //   _padding       [u32;1010] 4040 B
    //   ─────────────────────────────────
    //   TOTAL                     4096 B
    assert!(
        size_of::<u64>()      // frame_idx
        + 6 * size_of::<u32>()    // dt + adj + fence + n_atoms + gear_id + force_norm
        + 2 * size_of::<f64>()    // M1.2.18.5: external_work + potential_energy
        + 2 * size_of::<u32>()    // WP4: stream_id + schema_version
        + 1010 * size_of::<u32>() // tail padding
        == 4096
    );

    // WP4 — the header offsets are a THREE-way contract:
    //   1. this struct,
    //   2. the PRISM_ZSTR_HDR_OFF_* macros in src/cuda/zstr_kernels.cuh that
    //      the device slot-authority and audit kernels stamp through,
    //   3. the offline parser bin/prism_ghost_evidence_scan.rs, which derives
    //      the whole file stride from `n_atoms` at offset 20.
    // Drift in any one of them silently produces an unparseable ZSTR file, so
    // pin every stamped offset here rather than trusting declaration order.
    use std::mem::offset_of;
    assert!(offset_of!(ZstrFrameHeader, frame_idx) == 0);
    assert!(offset_of!(ZstrFrameHeader, dt) == 8);
    assert!(offset_of!(ZstrFrameHeader, adjudication_code) == 12);
    assert!(offset_of!(ZstrFrameHeader, completion_fence) == 16);
    assert!(offset_of!(ZstrFrameHeader, n_atoms) == 20);
    assert!(offset_of!(ZstrFrameHeader, gear_id) == 24);
    assert!(offset_of!(ZstrFrameHeader, force_norm) == 28);
    assert!(offset_of!(ZstrFrameHeader, external_work) == 32);
    assert!(offset_of!(ZstrFrameHeader, potential_energy) == 40);
    assert!(offset_of!(ZstrFrameHeader, stream_id) == 48);
    assert!(offset_of!(ZstrFrameHeader, schema_version) == 52);
    assert!(offset_of!(ZstrFrameHeader, _padding) == 56);
};

/// **WP4** — current ZSTR header schema version stamped by the device
/// slot-authority kernel. Bump only when a header field is added or its
/// meaning changes; the 4096-byte size and offsets 0..48 are frozen.
pub const ZSTR_HEADER_SCHEMA_VERSION: u32 = 1;

/// **WP4** — sentinel meaning "no slot claimed for this epoch". Mirrors
/// `PRISM_ZSTR_SLOT_NONE` in `src/cuda/zstr_kernels.cuh`.
pub const ZSTR_SLOT_NONE: u32 = 0xFFFF_FFFF;

/// **GEAR-PROVENANCE-001 (2026-08-02)** — sentinel meaning "G26 gear was not
/// measured for this frame". Mirrors `PRISM_ZSTR_GEAR_ID_UNMEASURED` in
/// `src/cuda/zstr_kernels.cuh`.
///
/// The header's `gear_id` uses the documented encoding 0 = 0.5 fs,
/// 1 = 2.0 fs, 2 = 4.0 fs (HMR), 3 = abort-trap. Every valid gear is therefore
/// a small integer, and there is no in-band way to say "unknown" — 0 is a
/// *legitimate* gear, so a placeholder 0 is indistinguishable offline from a
/// genuine high-resolution frame.
///
/// The V2 config supplies a hardcoded `gear_id: 0` ("Wave A default") while the
/// device-side `cruise->current_gear` never leaves the GPU. Stamping that
/// literal would assert 0.5 fs on every record of a run actually integrating at
/// 2.0 or 4.0 fs. Since the chronology's units are exactly what makes the
/// record stream scientifically usable, a wrong-but-plausible gear is worse
/// than an absent one: it silently rescales every downstream time axis.
///
/// Readers MUST treat this value as "resolution not established" and fall back
/// to the header `dt` field, which carries the real timestep in picoseconds.
/// When the gearbox publishes a measured gear to the host, stamp that instead.
pub const ZSTR_GEAR_ID_UNMEASURED: u32 = 0xFFFF_FFFF;

/// **WP4** — byte offsets inside the per-stream backpressure/conservation
/// status block. Mirrors `PRISM_ZSTR_BP_OFF_*` in `zstr_kernels.cuh`.
pub const ZSTR_BP_OFF_ERROR: usize = 0;
pub const ZSTR_BP_OFF_SLOT: usize = 4;
pub const ZSTR_BP_OFF_EPOCHS: usize = 8;
pub const ZSTR_BP_OFF_PRODUCED: usize = 16;
pub const ZSTR_BP_OFF_EVENTS: usize = 24;

/// **WP4** — backpressure error codes. Mirrors `PRISM_ZSTR_BP_ERR_*`.
pub const ZSTR_BP_ERR_NONE: u32 = 0;
pub const ZSTR_BP_ERR_SPIN_EXPIRED: u32 = 1;
pub const ZSTR_BP_ERR_NULL_ARG: u32 = 2;

/// **WP4** — device backpressure spin budget. Each iteration is a PCIe
/// read of pinned host memory plus a ~1 µs `__nanosleep`, so the budget is
/// approximately a microsecond count: 4_000_000 ≈ 4 s of tolerated drain
/// latency per epoch before the claim is refused. Overridable via
/// `PRISM_ZSTR_BACKPRESSURE_SPIN_LIMIT` for substrates with slower NVMe.
pub const ZSTR_DEFAULT_BACKPRESSURE_SPIN_LIMIT: u32 = 4_000_000;

/// Resolve the device backpressure spin budget, honouring the
/// `PRISM_ZSTR_BACKPRESSURE_SPIN_LIMIT` environment override. A value of 0
/// disables the bounded wait entirely and restores pre-WP4 overwrite
/// semantics; it is accepted only so a forensic A/B against the old
/// behavior remains reachable, and it is logged loudly.
pub fn resolve_backpressure_spin_limit() -> u32 {
    match std::env::var("PRISM_ZSTR_BACKPRESSURE_SPIN_LIMIT") {
        Ok(raw) => match raw.trim().parse::<u32>() {
            Ok(0) => {
                log::warn!(
                    "[ZSTR WP4] PRISM_ZSTR_BACKPRESSURE_SPIN_LIMIT=0 — device \
                     backpressure DISABLED; unpersisted records may be \
                     overwritten (pre-WP4 semantics)."
                );
                0
            }
            Ok(v) => v,
            Err(e) => {
                log::warn!(
                    "[ZSTR WP4] PRISM_ZSTR_BACKPRESSURE_SPIN_LIMIT={:?} unparseable \
                     ({}) — using default {}",
                    raw,
                    e,
                    ZSTR_DEFAULT_BACKPRESSURE_SPIN_LIMIT
                );
                ZSTR_DEFAULT_BACKPRESSURE_SPIN_LIMIT
            }
        },
        Err(_) => ZSTR_DEFAULT_BACKPRESSURE_SPIN_LIMIT,
    }
}

// ─── ZSTR Consumer Thread ────────────────────────────────────────────────────

/// Message passed from the ZSTR consumer thread to the caller when
/// the run completes.
///
/// **WP4** — this is now a conservation receipt, not a log line. The
/// identity the caller must be able to check is
///
/// ```text
///   expected == produced + backpressure_events
///   produced == persisted                       (lossless persistence)
///   duplicates == 0 && gaps == 0
///   terminal_error.is_none() && fsync_ok
/// ```
///
/// `frames_written` / `frames_dropped` are retained verbatim for the
/// pre-WP4 log-line callers; on a healthy run `frames_dropped` is 0.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZstrStats {
    /// Legacy alias for `persisted`.
    pub frames_written: u64,
    /// Legacy counter. WP4: `produced - persisted`, i.e. records the device
    /// staged and fenced that never reached the file. MUST be 0 on a
    /// healthy run. A fence timeout is NOT counted here — an unready slot
    /// is idleness, not loss.
    pub frames_dropped: u64,
    pub bytes_written: u64,

    // ── WP4 conservation terms ──────────────────────────────────────────
    /// Coordinate epochs the device actually executed (slot-authority
    /// kernel invocations). Read from the per-stream pinned control block.
    pub expected: u64,
    /// Epochs that claimed a slot, staged a payload and raised the fence.
    pub produced: u64,
    /// Records this consumer wrote to the ZSTR file and verified as a
    /// full-length write.
    pub persisted: u64,
    /// Records whose stamped device step was <= the previously persisted
    /// step (slot reuse without a fresh stamp, or a header regression).
    pub duplicates: u64,
    /// Missing expected observations between consecutive persisted records.
    pub gaps: u64,
    /// Explicit producer contract, never inferred from observed GCD.
    pub expected_step_cadence: Option<u64>,
    pub cadence_mismatches: u64,
    /// Device step stamped in the first persisted record.
    pub first_device_step: Option<u64>,
    /// Device step stamped in the last persisted record.
    pub last_device_step: Option<u64>,
    /// Maximum number of simultaneously READY slots observed. A value of
    /// `N_SLOTS` means the consumer was the bottleneck at least once and
    /// the device was applying backpressure.
    pub ring_high_water: u32,
    /// Claims refused by the device-side bounded wait.
    pub backpressure_events: u64,
    /// Device backpressure error code (`ZSTR_BP_ERR_*`); 0 == clean.
    pub backpressure_error: u32,
    /// Slot that tripped the bounded wait (valid iff `backpressure_error != 0`).
    pub backpressure_error_slot: u32,
    /// Non-finite `force_norm` reads (G29 Action-Recovery trap).
    pub g29_nonfinite_events: u64,
    /// Device step of the first G29 trap, if any.
    pub g29_first_device_step: Option<u64>,
    /// Header schema version observed in the last persisted record.
    pub observed_schema_version: u32,
    /// Per-slot / per-record byte stride actually written.
    pub frame_size: u64,
    /// Atom count the ring was allocated for.
    pub n_atoms: u64,
    /// `fsync(2)` on the Channel-A fd returned 0.
    pub fsync_ok: bool,
    /// Channel-A output path.
    pub output_path: std::path::PathBuf,
    /// First fatal persistence fault. Any value here means the ZSTR file
    /// is NOT a complete record of the run.
    pub terminal_error: Option<String>,
}

impl ZstrStats {
    /// Stats for a consumer that never reached its loop.
    fn aborted(output_path: std::path::PathBuf, reason: String) -> Self {
        Self {
            frames_written: 0,
            frames_dropped: 0,
            bytes_written: 0,
            expected: 0,
            produced: 0,
            persisted: 0,
            duplicates: 0,
            gaps: 0,
            expected_step_cadence: None,
            cadence_mismatches: 0,
            first_device_step: None,
            last_device_step: None,
            ring_high_water: 0,
            backpressure_events: 0,
            backpressure_error: ZSTR_BP_ERR_NONE,
            backpressure_error_slot: 0,
            g29_nonfinite_events: 0,
            g29_first_device_step: None,
            observed_schema_version: 0,
            frame_size: 0,
            n_atoms: 0,
            fsync_ok: false,
            output_path,
            terminal_error: Some(reason),
        }
    }

    /// True iff every WP4 conservation term is satisfied.
    pub fn conservation_pass(&self) -> bool {
        self.terminal_error.is_none()
            && self.fsync_ok
            && self.produced == self.persisted
            && self.expected == self.produced + self.backpressure_events
            && self.duplicates == 0
            && self.gaps == 0
            && self.expected_step_cadence.is_some()
            && self.cadence_mismatches == 0
            && self.backpressure_error == ZSTR_BP_ERR_NONE
    }
}

/// Spawn the io_uring + O_DIRECT ZSTR consumer thread.
///
/// **Wave 0 / #69 — Beast-Mode I/O.** Replaces the pwrite(2) path with
/// pure io_uring SQE submission.  No VFS read/write copy, no page
/// cache (O_DIRECT bypasses it; io_uring bypasses the buffered-IO
/// stack entirely).  Each ready ring slot is slammed straight into
/// the NVMe submission queue as an `IORING_OP_WRITE` SQE pointing at
/// the pinned-host slot bytes; the kernel's blk-mq layer hands them
/// to the device without an intermediate copy.
///
/// **WP4 — discovery, not prediction.** The pre-WP4 loop maintained its
/// own `frame_idx` counter and derived `slot = frame_idx % N_SLOTS`. The
/// device derived its slot from `(*d_step_counter) % n_slots` with the
/// counter SEEDED at `v2_trigger_step`, and the host counter advanced on
/// every loop iteration including fence timeouts. The two indices were
/// therefore phase-shifted from the first record and drifted further on
/// every timeout, and a timeout — a pure timing miss — was recorded as
/// `frames_dropped`, i.e. as a lost scientific record.
///
/// The thread now:
/// 1. Opens `output_path` with `O_DIRECT | O_CREAT | O_WRONLY`.
/// 2. Initialises a 32-entry `IoUring`.
/// 3. Loops:
///    a. Sweeps ALL `N_SLOTS` headers for `completion_fence == 1`.
///    b. Among the ready slots, selects the one with the LOWEST stamped
///       `frame_idx` (device step), so the file is in chronological order
///       regardless of the producer's slot permutation.
///    c. No ready slot ⇒ IDLE. Yield and rescan. Never a drop.
///    d. Writes the slot with one `IORING_OP_WRITE` SQE and requires
///       `cqe.result() == frame_size` EXACTLY. Short writes were
///       previously counted as successes and advanced the file offset by
///       a full frame, desynchronising the on-disk stride.
///    e. Reads + traps non-finite `force_norm` (G29).
///    f. Resets `force_norm` then `completion_fence` (release-ordered).
///    g. Tracks the last persisted device step for duplicate/gap detection.
/// 4. On `stop`: drains every remaining READY slot, verifies no slot is
///    still fenced, `fsync(2)`s, and returns the conservation receipt.
///
/// Channel B is intentionally not written here. Ghost persistence is owned by
/// `ghost_segment::GhostSegmentWriter` at synchronized engine chunk
/// boundaries. Keeping the legacy arguments avoids an ABI-wide caller change,
/// but non-`None` values are rejected and ignored so a second writer can never
/// race or truncate the canonical segment stream.
///
/// # Safety
/// `ring` and (if Some) `ghost_ring` must remain valid for the
/// lifetime of the returned thread.  The caller joins before dropping.

/// Exact pinned bytes at a synchronized, fully drained consumer boundary.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZstrRingContinuationV1 {
    pub schema_version: u32,
    pub n_atoms: usize,
    pub frame_size: usize,
    #[serde(with="crate::checkpoint_codec::bytes")]
    pub slots: Vec<u8>,
    #[serde(with="crate::checkpoint_codec::bytes")]
    pub control: Vec<u8>,
}
impl ZstrRing {
    /// # Safety
    /// Every CUDA stream touching this allocation must be synchronized and held;
    /// the sole host consumer must be paused after draining READY slots.
    pub unsafe fn checkpoint_quiescent(&self) -> anyhow::Result<ZstrRingContinuationV1> {
        anyhow::ensure!(
            self.alignment_ok && !self.pinned_ptr.is_null() && !self.control_ptr.is_null(),
            "ZSTR allocation invalid"
        );
        for slot in 0..Self::N_SLOTS {
            anyhow::ensure!(
                std::ptr::read_volatile(std::ptr::addr_of!(
                    (*self.header_ptr(slot)).completion_fence
                )) == 0,
                "ZSTR pending frame at checkpoint"
            );
        }
        let bytes = self
            .frame_size
            .checked_mul(Self::N_SLOTS)
            .ok_or_else(|| anyhow::anyhow!("ZSTR allocation length overflow"))?;
        Ok(ZstrRingContinuationV1 {
            schema_version: 1,
            n_atoms: self.n_atoms,
            frame_size: self.frame_size,
            slots: std::slice::from_raw_parts(self.pinned_ptr, bytes).to_vec(),
            control: std::slice::from_raw_parts(self.control_ptr, Self::CONTROL_BYTES).to_vec(),
        })
    }
    /// # Safety
    /// Fresh allocation has no GPU/host users yet; capture must rebind every old
    /// pointer to this allocation before replay. Serialized pointers are absent.
    pub unsafe fn restore_quiescent(&self, state: &ZstrRingContinuationV1) -> anyhow::Result<()> {
        anyhow::ensure!(
            state.schema_version == 1
                && state.n_atoms == self.n_atoms
                && state.frame_size == self.frame_size
                && state.slots.len()
                    == self
                        .frame_size
                        .checked_mul(Self::N_SLOTS)
                        .ok_or_else(|| anyhow::anyhow!("ZSTR size overflow"))?
                && state.control.len() == Self::CONTROL_BYTES
                && self.alignment_ok,
            "ZSTR continuation allocation contract differs"
        );
        for slot in 0..Self::N_SLOTS {
            let offset = slot * self.frame_size + offset_of_completion_fence();
            anyhow::ensure!(
                state.slots[offset..offset + 4] == [0; 4],
                "ZSTR saved consumer boundary is not drained"
            );
        }
        let active = u32::from_le_bytes(state.control[..4].try_into()?);
        anyhow::ensure!(
            active == ZSTR_SLOT_NONE || (active as usize) < Self::N_SLOTS,
            "ZSTR saved slot outside allocation"
        );
        std::ptr::copy_nonoverlapping(state.slots.as_ptr(), self.pinned_ptr, state.slots.len());
        std::ptr::copy_nonoverlapping(
            state.control.as_ptr(),
            self.control_ptr,
            state.control.len(),
        );
        std::sync::atomic::fence(Ordering::SeqCst);
        Ok(())
    }
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZstrContinuationV1 {
    pub schema_version: u32,
    pub checkpoint_generation: u64,
    pub output: crate::scientific_checkpoint::AppendCursor,
    pub ring: ZstrRingContinuationV1,
    pub stats: ZstrStats,
    pub idle_sweeps: u32,
    pub drain_confirmations: u32,
    pub last_ghost_slot_emitted: u32,
    pub ghost_records_written: u64,
    pub ghost_bytes_written: u64,
}

/// Re-derive durable chronology/fault counters from every committed record.
/// Coordinate/force bytes are bound separately by the full prefix SHA-256.
fn verify_zstr_checkpoint_records(
    path: &std::path::Path,
    state: &ZstrContinuationV1,
) -> anyhow::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    let mut first = None;
    let mut last = None;
    let mut duplicates = 0;
    let mut gaps = 0;
    let mut cadence_mismatches = 0;
    let mut g29 = 0;
    let mut first_g29 = None;
    let mut schema = 0;
    for index in 0..state.stats.persisted {
        file.seek(SeekFrom::Start(
            index
                .checked_mul(state.stats.frame_size)
                .ok_or_else(|| anyhow::anyhow!("ZSTR offset overflow"))?,
        ))?;
        let mut header = [0u8; 56];
        file.read_exact(&mut header)?;
        let step = u64::from_le_bytes(header[0..8].try_into()?);
        let fence = u32::from_le_bytes(header[16..20].try_into()?);
        let atoms = u32::from_le_bytes(header[20..24].try_into()?);
        let norm = f32::from_le_bytes(header[28..32].try_into()?);
        schema = u32::from_le_bytes(header[52..56].try_into()?);
        anyhow::ensure!(
            fence == 1 && atoms as u64 == state.stats.n_atoms,
            "ZSTR persisted header population/fence differs"
        );
        if let Some(previous) = last {
            if step <= previous {
                duplicates += 1;
            } else if let Some(cadence) = state.stats.expected_step_cadence {
                anyhow::ensure!(cadence > 0, "ZSTR zero cadence");
                let delta = step - previous;
                if delta % cadence != 0 {
                    cadence_mismatches += 1;
                } else {
                    gaps += delta / cadence - 1;
                }
            }
        }
        first.get_or_insert(step);
        last = Some(step);
        if !norm.is_finite() {
            g29 += 1;
            first_g29.get_or_insert(step);
        }
    }
    anyhow::ensure!(
        first == state.stats.first_device_step
            && last == state.stats.last_device_step
            && duplicates == state.stats.duplicates
            && gaps == state.stats.gaps
            && cadence_mismatches == state.stats.cadence_mismatches
            && g29 == state.stats.g29_nonfinite_events
            && first_g29 == state.stats.g29_first_device_step
            && schema == state.stats.observed_schema_version,
        "ZSTR saved host counters differ from committed native records"
    );
    Ok(())
}
enum ZstrCheckpointCommand {
    Pause {
        generation: u64,
        run_root: std::path::PathBuf,
        expected_epochs: u64,
        expected_produced: u64,
        reply: std::sync::mpsc::SyncSender<Result<ZstrContinuationV1, String>>,
    },
    Resume {
        generation: u64,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
}
pub struct ZstrCheckpointControl {
    sender: std::sync::mpsc::SyncSender<ZstrCheckpointCommand>,
}
impl ZstrCheckpointControl {
    /// # Safety
    /// Caller holds every GPU producer at a completed stream fence until this
    /// pause acknowledgement and the native capsule commit are complete.
    pub unsafe fn pause(
        &self,
        generation: u64,
        run_root: &std::path::Path,
        expected_epochs: u64,
        expected_produced: u64,
    ) -> Result<ZstrContinuationV1, String> {
        let (reply, rx) = std::sync::mpsc::sync_channel(1);
        self.sender
            .send(ZstrCheckpointCommand::Pause {
                generation,
                run_root: run_root.into(),
                expected_epochs,
                expected_produced,
                reply,
            })
            .map_err(|_| "ZSTR checkpoint owner disconnected".to_string())?;
        rx.recv()
            .map_err(|_| "ZSTR checkpoint acknowledgement missing".to_string())?
    }
    pub fn resume(&self, generation: u64) -> Result<(), String> {
        let (reply, rx) = std::sync::mpsc::sync_channel(1);
        self.sender
            .send(ZstrCheckpointCommand::Resume { generation, reply })
            .map_err(|_| "ZSTR resume owner disconnected".to_string())?;
        rx.recv()
            .map_err(|_| "ZSTR resume acknowledgement missing".to_string())?
    }
}
struct ZstrStartup {
    reply: Option<std::sync::mpsc::SyncSender<Result<(), String>>>,
}
impl ZstrStartup {
    fn ready(&mut self) {
        if let Some(reply) = self.reply.take() {
            let _ = reply.send(Ok(()));
        }
    }
}
impl Drop for ZstrStartup {
    fn drop(&mut self) {
        if let Some(reply) = self.reply.take() {
            let _ = reply.send(Err(
                "ZSTR consumer failed before durable writer admission".into()
            ));
        }
    }
}
/// Rebuild owner handles, preserve the output prefix and start the actual
/// io_uring/O_DIRECT consumer. The call waits for real consumer admission.
/// # Safety
/// Ring allocation and captured aliases have no concurrent CUDA/host users.
pub unsafe fn spawn_zstr_consumer_continuable(
    ring: Arc<ZstrRing>,
    output_path: std::path::PathBuf,
    stop: Arc<AtomicBool>,
    expected_step_cadence: Option<u64>,
    resume: Option<(std::path::PathBuf, ZstrContinuationV1)>,
) -> Result<(JoinHandle<ZstrStats>, ZstrCheckpointControl), String> {
    if stop.load(Ordering::Acquire) {
        return Err("ZSTR stop flag already set at admission".into());
    }
    if let Some((root, state)) = &resume {
        if state.schema_version != 1
            || root.join(&state.output.relative_path) != output_path
            || state.stats.expected_step_cadence != expected_step_cadence
            || state.stats.terminal_error.is_some()
            || !state.stats.fsync_ok
            || state.stats.bytes_written != state.output.committed_bytes
            || state
                .stats
                .persisted
                .checked_mul(state.ring.frame_size as u64)
                != Some(state.stats.bytes_written)
            || state.stats.frame_size != state.ring.frame_size as u64
            || state.stats.n_atoms != state.ring.n_atoms as u64
            || state.last_ghost_slot_emitted != 0
            || state.ghost_records_written != 0
            || state.ghost_bytes_written != 0
        {
            return Err("ZSTR continuation identity/counter contract differs".into());
        }
        ring.restore_quiescent(&state.ring)
            .map_err(|e| e.to_string())?;
        if ring.device_epochs() != state.stats.expected
            || ring.device_produced() != state.stats.produced
            || state.stats.produced != state.stats.persisted
            || ring.device_backpressure_events() != state.stats.backpressure_events
            || ring.backpressure_error() != state.stats.backpressure_error
        {
            return Err("ZSTR restored device/consumer counters differ".into());
        }
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let (ready, started) = std::sync::mpsc::sync_channel(1);
    let handle = spawn_zstr_consumer_impl(
        ring,
        output_path,
        stop.clone(),
        expected_step_cadence,
        None,
        None,
        rx,
        resume,
        Some(ready),
    );
    match started.recv() {
        Ok(Ok(())) => Ok((handle, ZstrCheckpointControl { sender: tx })),
        result => {
            stop.store(true, Ordering::Release);
            let _ = handle.join();
            Err(format!("ZSTR writer admission failed: {result:?}"))
        }
    }
}

pub fn spawn_zstr_consumer(
    ring: Arc<ZstrRing>,
    output_path: std::path::PathBuf,
    stop: Arc<AtomicBool>,
    expected_step_cadence: Option<u64>,
    ghost_ring: Option<Arc<crate::ghost_tile::GhostTileRing>>,
    ghost_output_path: Option<std::path::PathBuf>,
) -> JoinHandle<ZstrStats> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    drop(tx);
    spawn_zstr_consumer_impl(
        ring,
        output_path,
        stop,
        expected_step_cadence,
        ghost_ring,
        ghost_output_path,
        rx,
        None,
        None,
    )
}
fn spawn_zstr_consumer_impl(
    ring: Arc<ZstrRing>,
    output_path: std::path::PathBuf,
    stop: Arc<AtomicBool>,
    expected_step_cadence: Option<u64>,
    ghost_ring: Option<Arc<crate::ghost_tile::GhostTileRing>>,
    ghost_output_path: Option<std::path::PathBuf>,
    checkpoint_rx: std::sync::mpsc::Receiver<ZstrCheckpointCommand>,
    resume: Option<(std::path::PathBuf, ZstrContinuationV1)>,
    startup_reply: Option<std::sync::mpsc::SyncSender<Result<(), String>>>,
) -> JoinHandle<ZstrStats> {
    std::thread::Builder::new()
        .name(format!("zstr-consumer"))
        .spawn(move || {
            let mut startup = ZstrStartup {
                reply: startup_reply,
            };
            use io_uring::{opcode, types, IoUring};
            use std::os::unix::fs::OpenOptionsExt;

            let stats_path = output_path.clone();
            if expected_step_cadence == Some(0) {
                return ZstrStats::aborted(
                    stats_path,
                    "ZSTR expected step cadence must be explicit and positive".into(),
                );
            }

            if ghost_ring.is_some() || ghost_output_path.is_some() {
                log::error!(
                    "[ZSTR consumer] legacy Channel-B writer request rejected; \
                     GhostSegmentWriter is the sole Ghost artifact authority"
                );
            }
            // Shadow the retired inputs. The remaining historical drain block
            // is thereby unreachable while its Channel-A mechanics stay
            // untouched in this surgical repair.
            let ghost_ring: Option<Arc<crate::ghost_tile::GhostTileRing>> = None;
            let ghost_output_path: Option<std::path::PathBuf> = None;

            // G21 gate enforcement — consumer must not proceed if alignment failed.
            if !ring.alignment_ok {
                log::error!(
                    "[ZSTR consumer] G21_ALIGNMENT_PASS not met — \
                     pinned buffer not 4096-aligned.  O_DIRECT writes \
                     will fault.  Aborting consumer."
                );
                return ZstrStats::aborted(
                    stats_path,
                    "ZSTR_G21_ALIGNMENT_FAIL: pinned buffer not 4096-aligned".to_string(),
                );
            }

            let file = (|| -> std::io::Result<std::fs::File> {
                if let Some((root, state)) = &resume {
                    crate::scientific_checkpoint::reopen_append_file(root, &state.output).map_err(
                        |e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
                    )?;
                    verify_zstr_checkpoint_records(&output_path, state).map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                    })?;
                    std::fs::OpenOptions::new()
                        .write(true)
                        .custom_flags(libc::O_DIRECT | libc::O_NOFOLLOW | libc::O_CLOEXEC)
                        .open(&output_path)
                } else {
                    std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .custom_flags(libc::O_DIRECT | libc::O_NOFOLLOW | libc::O_CLOEXEC)
                        .open(&output_path)
                }
            })();

            let file = match file {
                Ok(f) => f,
                Err(e) => {
                    log::error!("[ZSTR consumer] Cannot open {:?}: {}", output_path, e);
                    return ZstrStats::aborted(
                        stats_path,
                        format!("ZSTR_OPEN_FAILED: {:?}: {}", output_path, e),
                    );
                }
            };

            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();

            // **Amendment 3.14 / v9D' Sector-Lock** — Channel-B ghost output file.
            // GhostTileFrame is now 4096-byte sector-aligned, so this fd is
            // opened with `O_DIRECT | O_DSYNC`: every per-record SQE write
            // bypasses the page cache and lands on a unique NVMe physical
            // sector (no read-modify-write penalty on the Samsung 990 PRO
            // controller under PCIe Gen5).
            let ghost_state: Option<(std::fs::File, i32)> = ghost_output_path
                .as_ref()
                .filter(|_| ghost_ring.is_some())
                .and_then(|p| {
                    use std::os::unix::fs::OpenOptionsExt;
                    match std::fs::OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .custom_flags(libc::O_DIRECT | libc::O_DSYNC)
                        .open(p)
                    {
                        Ok(f) => {
                            let raw = f.as_raw_fd();
                            log::info!(
                                "[AMS Amendment-3.14] Channel-B fd={} opened \
                                 (O_DIRECT | O_DSYNC; 4096 B stride sector-locked) \
                                 → {:?}",
                                raw,
                                p
                            );
                            Some((f, raw))
                        }
                        Err(e) => {
                            log::warn!(
                                "[AMS Amendment-3.14] Cannot open ghost output {:?}: {} \
                                 — Channel-B drain disabled, teardown-only flush",
                                p,
                                e
                            );
                            None
                        }
                    }
                });
            let ghost_fd: i32 = ghost_state.as_ref().map(|(_, fd)| *fd).unwrap_or(-1);
            let mut last_ghost_slot_emitted: u32 = 0;
            let mut ghost_records_written: u64 = 0;
            let mut ghost_bytes_written: u64 = 0;
            const GHOST_SAFETY_LAG: u32 = 1;
            const GHOST_RECORD_STRIDE: usize = 4096;

            // Amendment 3.14 / v9D' — io_uring instance.  Bumped to 32
            // entries (was 16) to absorb the high-frequency 6-step
            // chunking emission rate without SQ saturation.  At most 5
            // ring slots inflight from the GPU side; the remaining 27
            // SQEs are queue headroom for ZSTR + ghost burst overlap.
            // Build with default flags (no SQPOLL — submit/wait
            // synchronous-per-frame contract preserves slot recycle
            // ordering).
            let mut uring: IoUring = match IoUring::new(32) {
                Ok(u) => u,
                Err(e) => {
                    log::error!(
                        "[ZSTR consumer] io_uring init failed: {} \
                         — Check kernel build for CONFIG_IO_URING=y.",
                        e
                    );
                    return ZstrStats::aborted(
                        stats_path,
                        format!("ZSTR_IO_URING_INIT_FAILED: {}", e),
                    );
                }
            };
            log::info!(
                "[ZSTR consumer] io_uring initialised: 32 SQEs, O_DIRECT \
                 fd={} → {:?}",
                fd,
                output_path
            );

            let frame_size = ring.frame_size;
            let n_slots = ZstrRing::N_SLOTS;

            let mut persisted: u64 = 0;
            let mut bytes_written: u64 = 0;
            let mut duplicates: u64 = 0;
            let mut gaps: u64 = 0;
            let mut cadence_mismatches: u64 = 0;
            let mut first_device_step: Option<u64> = None;
            let mut last_device_step: Option<u64> = None;
            let mut ring_high_water: u32 = 0;
            let mut g29_nonfinite_events: u64 = 0;
            let mut g29_first_device_step: Option<u64> = None;
            let mut observed_schema_version: u32 = 0;
            let mut terminal_error: Option<String> = None;

            // Number of consecutive empty sweeps required, after `stop`, before
            // declaring the ring drained.  Successive sweeps separated by a
            // settle sleep close the window where the device fences a slot
            // between the sweep and the break.  Any slot that becomes ready
            // during the settle resets the counter and is drained normally.
            const DRAIN_CONFIRM_SWEEPS: u32 = 3;
            const DRAIN_SETTLE: std::time::Duration = std::time::Duration::from_millis(2);
            // Idle backoff: spin-yield while hot, then sleep so an idle
            // consumer does not peg a core for the whole campaign.
            const IDLE_YIELD_SWEEPS: u32 = 4096;
            const IDLE_SLEEP: std::time::Duration = std::time::Duration::from_micros(50);
            // Bounded SQ-push retry.  A full SQ is a transient condition, not
            // a reason to abandon a record.
            const SQ_PUSH_ATTEMPTS: u32 = 64;

            let mut idle_sweeps: u32 = 0;
            let mut drain_confirmations: u32 = 0;

            let mut last_checkpoint_generation =
                resume.as_ref().map(|(_, s)| s.checkpoint_generation);
            if let Some((_, state)) = &resume {
                persisted = state.stats.persisted;
                bytes_written = state.stats.bytes_written;
                duplicates = state.stats.duplicates;
                gaps = state.stats.gaps;
                cadence_mismatches = state.stats.cadence_mismatches;
                first_device_step = state.stats.first_device_step;
                last_device_step = state.stats.last_device_step;
                ring_high_water = state.stats.ring_high_water;
                g29_nonfinite_events = state.stats.g29_nonfinite_events;
                g29_first_device_step = state.stats.g29_first_device_step;
                observed_schema_version = state.stats.observed_schema_version;
                idle_sweeps = state.idle_sweeps;
                drain_confirmations = state.drain_confirmations;
                last_ghost_slot_emitted = state.last_ghost_slot_emitted;
                ghost_records_written = state.ghost_records_written;
                ghost_bytes_written = state.ghost_bytes_written;
            }
            startup.ready();
            let mut pending_checkpoint: Option<ZstrCheckpointCommand> = None;
            'consume: loop {
                let stopping = stop.load(Ordering::Acquire);
                if pending_checkpoint.is_none() {
                    if let Ok(command) = checkpoint_rx.try_recv() {
                        match command {
                            command @ ZstrCheckpointCommand::Pause { .. } => {
                                pending_checkpoint = Some(command)
                            }
                            ZstrCheckpointCommand::Resume { reply, .. } => {
                                let _ = reply.send(Err("ZSTR was not paused".into()));
                            }
                        }
                    }
                }

                // ── 1. Discovery sweep: which slots are READY? ──────────────
                let mut ready_count: u32 = 0;
                let mut chosen: Option<(u64, usize)> = None;
                for slot in 0..n_slots {
                    let hdr = unsafe { ring.header_ptr(slot) };
                    let fence = unsafe {
                        std::ptr::read_volatile(std::ptr::addr_of!((*hdr).completion_fence))
                    };
                    if fence != 1 {
                        continue;
                    }
                    ready_count += 1;
                    let step =
                        unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*hdr).frame_idx)) };
                    // ── 2. Lowest stamped device step wins.  The producer's
                    //    slot permutation is irrelevant to file order.
                    match chosen {
                        Some((best_step, _)) if best_step <= step => {}
                        _ => chosen = Some((step, slot)),
                    }
                }
                if ready_count > ring_high_water {
                    ring_high_water = ready_count;
                }

                let (device_step, slot) = match chosen {
                    Some(v) => {
                        idle_sweeps = 0;
                        drain_confirmations = 0;
                        v
                    }
                    None => {
                        if let Some(ZstrCheckpointCommand::Pause {
                            generation,
                            run_root,
                            expected_epochs,
                            expected_produced,
                            reply,
                        }) = pending_checkpoint.take()
                        {
                            let result = (|| -> anyhow::Result<ZstrContinuationV1> {
                                use std::os::unix::fs::MetadataExt;
                                anyhow::ensure!(
                                    !stopping
                                        && terminal_error.is_none()
                                        && last_checkpoint_generation
                                            .is_none_or(|previous| generation > previous),
                                    "ZSTR checkpoint lifecycle/generation invalid"
                                );
                                anyhow::ensure!(
                                    uring.submission().is_empty() && uring.completion().is_empty(),
                                    "ZSTR checkpoint has unconsumed I/O"
                                );
                                anyhow::ensure!(
                                    ring.device_epochs() == expected_epochs
                                        && ring.device_produced() == expected_produced
                                        && persisted == expected_produced
                                        && expected_epochs
                                            == expected_produced
                                                .checked_add(ring.device_backpressure_events())
                                                .ok_or_else(|| anyhow::anyhow!(
                                                    "ZSTR counter overflow"
                                                ))?
                                        && ring.backpressure_error() == ZSTR_BP_ERR_NONE,
                                    "ZSTR producer/consumer checkpoint conservation failed"
                                );
                                anyhow::ensure!(
                                    last_ghost_slot_emitted == 0
                                        && ghost_records_written == 0
                                        && ghost_bytes_written == 0,
                                    "retired ZSTR Ghost owner acquired state"
                                );
                                file.sync_all()?;
                                let reader = std::fs::OpenOptions::new()
                                    .read(true)
                                    .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                                    .open(&output_path)?;
                                let a = file.metadata()?;
                                let b = reader.metadata()?;
                                anyhow::ensure!(
                                    a.dev() == b.dev() && a.ino() == b.ino(),
                                    "ZSTR output descriptor identity changed"
                                );
                                let relative = output_path
                                    .strip_prefix(&run_root)?
                                    .to_str()
                                    .ok_or_else(|| anyhow::anyhow!("ZSTR output encoding"))?;
                                let output = crate::scientific_checkpoint::seal_append_file(
                                    &reader, relative,
                                )?;
                                anyhow::ensure!(
                                    output.committed_bytes == bytes_written
                                        && persisted.checked_mul(frame_size as u64)
                                            == Some(bytes_written),
                                    "ZSTR committed byte population mismatch"
                                );
                                std::fs::File::open(
                                    output_path
                                        .parent()
                                        .ok_or_else(|| anyhow::anyhow!("ZSTR output parent"))?,
                                )?
                                .sync_all()?;
                                let ring_state = unsafe { ring.checkpoint_quiescent()? };
                                anyhow::ensure!(
                                    ring.device_epochs() == expected_epochs
                                        && ring.device_produced() == expected_produced,
                                    "ZSTR producer advanced across checkpoint"
                                );
                                let stats = ZstrStats {
                                    frames_written: persisted,
                                    frames_dropped: 0,
                                    bytes_written,
                                    expected: expected_epochs,
                                    produced: expected_produced,
                                    persisted,
                                    duplicates,
                                    gaps,
                                    expected_step_cadence,
                                    cadence_mismatches,
                                    first_device_step,
                                    last_device_step,
                                    ring_high_water,
                                    backpressure_events: ring.device_backpressure_events(),
                                    backpressure_error: ring.backpressure_error(),
                                    backpressure_error_slot: ring.backpressure_error_slot(),
                                    g29_nonfinite_events,
                                    g29_first_device_step,
                                    observed_schema_version,
                                    frame_size: frame_size as u64,
                                    n_atoms: ring.n_atoms as u64,
                                    fsync_ok: true,
                                    output_path: stats_path.clone(),
                                    terminal_error: None,
                                };
                                Ok(ZstrContinuationV1 {
                                    schema_version: 1,
                                    checkpoint_generation: generation,
                                    output,
                                    ring: ring_state,
                                    stats,
                                    idle_sweeps,
                                    drain_confirmations,
                                    last_ghost_slot_emitted,
                                    ghost_records_written,
                                    ghost_bytes_written,
                                })
                            })();
                            match result {
                                Err(error) => {
                                    let message = error.to_string();
                                    let _ = reply.send(Err(message.clone()));
                                    terminal_error = Some(message);
                                    break 'consume;
                                }
                                Ok(state) => {
                                    if reply.send(Ok(state)).is_err() {
                                        terminal_error =
                                            Some("ZSTR checkpoint receiver disconnected".into());
                                        break 'consume;
                                    }
                                }
                            }
                            last_checkpoint_generation = Some(generation);
                            loop {
                                if stop.load(Ordering::Acquire) {
                                    terminal_error =
                                        Some("ZSTR stopped while checkpoint held".into());
                                    break 'consume;
                                }
                                match checkpoint_rx
                                    .recv_timeout(std::time::Duration::from_millis(50))
                                {
                                    Ok(ZstrCheckpointCommand::Resume {
                                        generation: requested,
                                        reply,
                                    }) => {
                                        if requested != generation {
                                            let _ = reply.send(Err(
                                                "ZSTR resume generation mismatch".into(),
                                            ));
                                            continue;
                                        }
                                        if reply.send(Ok(())).is_err() {
                                            terminal_error =
                                                Some("ZSTR resume receiver disconnected".into());
                                            break 'consume;
                                        }
                                        break;
                                    }
                                    Ok(ZstrCheckpointCommand::Pause { reply, .. }) => {
                                        let _ = reply.send(Err("ZSTR already paused".into()));
                                    }
                                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                                        terminal_error = Some(
                                            "ZSTR checkpoint owner disappeared while held".into(),
                                        );
                                        break 'consume;
                                    }
                                }
                            }
                            continue;
                        }
                        // ── 6. No ready slot is NOT a drop. It is idle. ─────
                        if stopping {
                            drain_confirmations += 1;
                            if drain_confirmations >= DRAIN_CONFIRM_SWEEPS {
                                break 'consume;
                            }
                            std::thread::sleep(DRAIN_SETTLE);
                            continue;
                        }
                        idle_sweeps = idle_sweeps.saturating_add(1);
                        if idle_sweeps < IDLE_YIELD_SWEEPS {
                            std::thread::yield_now();
                        } else {
                            std::thread::sleep(IDLE_SLEEP);
                        }
                        continue;
                    }
                };

                // ── 3. Persist.  io_uring zero-copy submit:
                //      pinned slot ptr → IORING_OP_WRITE SQE → NVMe queue.
                let slot_ptr = unsafe { ring.pinned_ptr.add(slot * frame_size) } as *const u8;
                let offset_i64 = (persisted * frame_size as u64) as i64;

                let write_e = opcode::Write::new(types::Fd(fd), slot_ptr, frame_size as u32)
                    .offset(offset_i64)
                    .build()
                    .user_data(device_step);

                // ── 8. SQ-push failure retries the SAME slot.  Submit what is
                //    already queued to make room, reap, retry.  Never a drop.
                let mut pushed = false;
                for attempt in 0..SQ_PUSH_ATTEMPTS {
                    if unsafe { uring.submission().push(&write_e) }.is_ok() {
                        pushed = true;
                        break;
                    }
                    match uring.submit() {
                        Ok(_) => {}
                        Err(e) => {
                            terminal_error = Some(format!(
                                "ZSTR_IO_SUBMIT_FAILED (SQ drain, device_step={} slot={} \
                                 attempt={}): {}",
                                device_step, slot, attempt, e
                            ));
                            break;
                        }
                    }
                    // Any CQE reaped here belongs to a write we already
                    // accounted for; the synchronous submit_and_wait(1)
                    // contract means the queue should be empty at this point.
                    let mut stray = 0u32;
                    while uring.completion().next().is_some() {
                        stray += 1;
                    }
                    if stray > 0 {
                        terminal_error = Some(format!(
                            "ZSTR_IO_UNACCOUNTED_CQE: {} stray completions reaped while \
                             draining a full SQ at device_step={} slot={}",
                            stray, device_step, slot
                        ));
                        break;
                    }
                    std::thread::yield_now();
                }
                if terminal_error.is_some() {
                    break 'consume;
                }
                if !pushed {
                    terminal_error = Some(format!(
                        "ZSTR_IO_SQ_FULL: could not enqueue device_step={} slot={} after \
                         {} attempts",
                        device_step, slot, SQ_PUSH_ATTEMPTS
                    ));
                    break 'consume;
                }

                // ── 9. Every I/O fault below is FATAL, not a warn-and-continue.
                if let Err(e) = uring.submit_and_wait(1) {
                    terminal_error = Some(format!(
                        "ZSTR_IO_SUBMIT_AND_WAIT_FAILED (device_step={} slot={}): {}",
                        device_step, slot, e
                    ));
                    break 'consume;
                }
                let written: i32 = match uring.completion().next() {
                    Some(cqe) => cqe.result(),
                    None => {
                        terminal_error = Some(format!(
                            "ZSTR_IO_MISSING_CQE: submit_and_wait returned but no completion \
                             present (device_step={} slot={})",
                            device_step, slot
                        ));
                        break 'consume;
                    }
                };
                if written < 0 {
                    terminal_error = Some(format!(
                        "ZSTR_IO_WRITE_FAILED (device_step={} slot={}): cqe.result={} \
                         (negative ⇒ -errno)",
                        device_step, slot, written
                    ));
                    break 'consume;
                }
                // ── 5. A short write is NOT a success.  Counting it as one
                //    advanced the file offset by a full frame and desynced the
                //    on-disk stride for every subsequent record.
                if written as usize != frame_size {
                    terminal_error = Some(format!(
                        "ZSTR_IO_SHORT_WRITE (device_step={} slot={}): wrote {} of {} bytes",
                        device_step, slot, written, frame_size
                    ));
                    break 'consume;
                }

                // ── G29 — Action-Recovery Zero-Trust trap.  Read the
                //    sqrt'd L2 norm written by zstr_force_norm_sqrt_kernel.
                //    Non-finite ⇒ ASC produced a non-physical force.  The
                //    bytes ARE on disk and are legitimate evidence of the
                //    fault, so the record is kept and counted; the campaign
                //    abort authority belongs to the orchestrator, which reads
                //    `g29_nonfinite_events` from this receipt.
                let (force_norm_val, adjudication_code, schema_version) = unsafe {
                    let hdr = ring.header_ptr(slot);
                    (
                        std::ptr::read_volatile(std::ptr::addr_of!((*hdr).force_norm)),
                        std::ptr::read_volatile(std::ptr::addr_of!((*hdr).adjudication_code)),
                        std::ptr::read_volatile(std::ptr::addr_of!((*hdr).schema_version)),
                    )
                };
                observed_schema_version = schema_version;
                if !force_norm_val.is_finite() {
                    g29_nonfinite_events += 1;
                    if g29_first_device_step.is_none() {
                        g29_first_device_step = Some(device_step);
                    }
                    log::error!(
                        "[ZSTR G29 TRAP] device_step={} slot={} force_norm={:?} non-finite — \
                         Action-Recovery Invariant violated; ASC steered into NaN/Inf. \
                         Record persisted as fault evidence. Operator must abort campaign.",
                        device_step,
                        slot,
                        force_norm_val
                    );
                }

                // ── 4. Reset force_norm then completion_fence, returning the
                //    slot to the GPU's writable pool.  Zeroing force_norm is
                //    mandatory: the next force_stage kernel atomic-adds Σ‖F‖²
                //    INTO this same field expecting it to start at 0.
                unsafe {
                    let hdr_mut = ring.header_ptr(slot);
                    let fence_ptr = std::ptr::addr_of_mut!((*hdr_mut).completion_fence);
                    let force_norm_ptr = std::ptr::addr_of_mut!((*hdr_mut).force_norm);
                    std::ptr::write_volatile(force_norm_ptr, 0.0f32);
                    // Force fence write to land AFTER the force_norm reset
                    // so a racing GPU launch on this slot never observes
                    // fence=0 with a stale non-zero force_norm.
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                    std::ptr::write_volatile(fence_ptr, 0u32);
                }

                // ── 5. Chronology bookkeeping.  `last_device_step` is the last
                //    step PERSISTED, so duplicates (slot reuse without a fresh
                //    stamp) and gaps (epochs the device never produced) are
                //    both detectable from the file alone.
                if let Some(prev) = last_device_step {
                    if device_step <= prev {
                        duplicates += 1;
                        log::warn!(
                            "[ZSTR WP4] duplicate/regressed device_step={} (previous={}) \
                             slot={} — header was not re-stamped for this epoch",
                            device_step,
                            prev,
                            slot
                        );
                    } else {
                        let delta = device_step - prev;
                        if let Some(cadence) = expected_step_cadence {
                            if delta % cadence != 0 {
                                cadence_mismatches += 1;
                            } else {
                                gaps += delta / cadence - 1;
                            }
                        }
                    }
                }
                if first_device_step.is_none() {
                    first_device_step = Some(device_step);
                }
                last_device_step = Some(device_step);

                persisted += 1;
                bytes_written += frame_size as u64;

                if persisted % 1000 == 1 {
                    log::info!(
                        "[ZSTR] device_step={} slot={} adj_code={} ‖F‖₂={:.6e} \
                         persisted={} ready_high_water={} written={} MB",
                        device_step,
                        slot,
                        adjudication_code,
                        force_norm_val,
                        persisted,
                        ring_high_water,
                        bytes_written >> 20,
                    );
                }

                // ── M1.2.20.C-H / T23 — Channel-B Ghost drain ──
                // Retired: `ghost_ring` is shadowed to None above, so this
                // block is unreachable. Kept verbatim (not deleted) because
                // GhostSegmentWriter owns Ghost persistence and the historical
                // mechanics remain the reference for that ownership boundary.
                if let (Some(ghost), Some(_)) = (ghost_ring.as_ref(), ghost_state.as_ref()) {
                    let counter_now = ghost.n_frames_written().min(ghost.max_records);
                    let safe_target = counter_now.saturating_sub(GHOST_SAFETY_LAG);
                    while last_ghost_slot_emitted < safe_target {
                        let slot_g = last_ghost_slot_emitted;
                        let src_off = crate::ghost_tile::GhostTileRing::COUNTER_SECTOR_BYTES
                            + (slot_g as usize) * GHOST_RECORD_STRIDE;
                        let dst_off = ghost_records_written * GHOST_RECORD_STRIDE as u64;
                        let src_ptr = unsafe { ghost.host_base.add(src_off) } as *const u8;
                        let g_write = opcode::Write::new(
                            types::Fd(ghost_fd),
                            src_ptr,
                            GHOST_RECORD_STRIDE as u32,
                        )
                        .offset(dst_off as i64)
                        .build()
                        .user_data(0xB000_0000_0000_0000u64 | slot_g as u64);
                        let g_push = unsafe { uring.submission().push(&g_write) };
                        if g_push.is_err() {
                            // SQ full — break and let next outer iter retry.
                            break;
                        }
                        let g_submit = uring.submit_and_wait(1);
                        let g_result: i32 = match g_submit {
                            Ok(_) => uring.completion().next().map(|c| c.result()).unwrap_or(-1),
                            Err(_) => -1,
                        };
                        if g_result < 0 {
                            log::warn!(
                                "[AMS T23] ghost slot {} write failed cqe.result={}; \
                                 dropping",
                                slot_g,
                                g_result
                            );
                            // Skip this slot but advance the cursor so we
                            // don't loop forever on a broken record.
                        } else {
                            ghost_records_written += 1;
                            ghost_bytes_written += g_result as u64;
                        }
                        last_ghost_slot_emitted += 1;
                    }
                }
            }

            // ── 7. Post-drain assertion: no slot may still hold a fenced,
            //    unpersisted record.  A stranded slot means the loop exited
            //    early (terminal error) or the device fenced after teardown.
            {
                let mut stranded: Vec<(usize, u64)> = Vec::new();
                for slot in 0..n_slots {
                    let hdr = unsafe { ring.header_ptr(slot) };
                    let fence = unsafe {
                        std::ptr::read_volatile(std::ptr::addr_of!((*hdr).completion_fence))
                    };
                    if fence == 1 {
                        let step = unsafe {
                            std::ptr::read_volatile(std::ptr::addr_of!((*hdr).frame_idx))
                        };
                        stranded.push((slot, step));
                    }
                }
                if !stranded.is_empty() {
                    let detail = stranded
                        .iter()
                        .map(|(s, step)| format!("slot{}@step{}", s, step))
                        .collect::<Vec<_>>()
                        .join(",");
                    log::error!(
                        "[ZSTR WP4] {} slot(s) still fenced at teardown: {} — records \
                         staged by the device were never persisted",
                        stranded.len(),
                        detail
                    );
                    if terminal_error.is_none() {
                        terminal_error = Some(format!(
                            "ZSTR_STRANDED_SLOTS: {} unpersisted fenced slot(s): {}",
                            stranded.len(),
                            detail
                        ));
                    }
                }
            }

            // **M1.2.20.C-H / T23** — Drain any remaining Channel-B
            // slots that the in-loop polling missed because of the
            // GHOST_SAFETY_LAG.  Retired alongside the in-loop block above;
            // unreachable while `ghost_ring` is shadowed to None.
            if let (Some(ghost), Some((_, _))) = (ghost_ring.as_ref(), ghost_state.as_ref()) {
                let counter_final = ghost.n_frames_written().min(ghost.max_records);
                while last_ghost_slot_emitted < counter_final {
                    let slot_g = last_ghost_slot_emitted;
                    let src_off = crate::ghost_tile::GhostTileRing::COUNTER_SECTOR_BYTES
                        + (slot_g as usize) * GHOST_RECORD_STRIDE;
                    let dst_off = ghost_records_written * GHOST_RECORD_STRIDE as u64;
                    let src_ptr = unsafe { ghost.host_base.add(src_off) } as *const u8;
                    let entry = opcode::Write::new(
                        types::Fd(ghost_fd),
                        src_ptr,
                        GHOST_RECORD_STRIDE as u32,
                    )
                    .offset(dst_off as i64)
                    .build()
                    .user_data(0xB000_0000_0000_0000u64 | slot_g as u64);
                    if unsafe { uring.submission().push(&entry) }.is_err() {
                        break;
                    }
                    let _ = uring.submit_and_wait(1);
                    if let Some(c) = uring.completion().next() {
                        let r = c.result();
                        if r > 0 {
                            ghost_records_written += 1;
                            ghost_bytes_written += r as u64;
                        }
                    }
                    last_ghost_slot_emitted += 1;
                }
                if ghost_fd >= 0 {
                    unsafe {
                        libc::fdatasync(ghost_fd);
                    }
                }
                log::info!(
                    "[AMS T23] Channel-B io_uring drain summary: \
                     records_written={} bytes_written={} (~{} MB) ghost_path={:?}",
                    ghost_records_written,
                    ghost_bytes_written,
                    ghost_bytes_written >> 20,
                    ghost_output_path
                );
            }

            // Amendment 3.21.1 §2.1 — Flush-before-Join invariant.
            // fsync(2) on BOTH Channel-A and Channel-B fds so the Samsung
            // 990 PRO controller commits its volatile DRAM-resident
            // metadata + data sectors to NAND before this thread joins.
            // fsync upgraded from fdatasync to also flush inode metadata
            // (file size, mtime) — required so the post-run validator's
            // stat() and JSON parsers see a consistent on-disk header.
            // WP4: an fsync failure is FATAL — an unflushed file is not a
            // durable record, and the pre-WP4 warn-and-continue let the run
            // report success over data that may never have reached NAND.
            let mut fsync_ok = false;
            unsafe {
                let rc_a = libc::fsync(fd);
                if rc_a != 0 {
                    let errno = *libc::__errno_location();
                    log::error!(
                        "[ZSTR teardown] Channel-A fsync(fd={}) returned {} (errno={})",
                        fd,
                        rc_a,
                        errno
                    );
                    if terminal_error.is_none() {
                        terminal_error = Some(format!(
                            "ZSTR_FSYNC_FAILED: fsync(fd={}) rc={} errno={}",
                            fd, rc_a, errno
                        ));
                    }
                } else {
                    fsync_ok = true;
                    log::info!("[ZSTR teardown] ✓ Channel-A fsync(fd={}) committed", fd);
                }
                if ghost_fd >= 0 {
                    let rc_b = libc::fsync(ghost_fd);
                    if rc_b != 0 {
                        log::warn!(
                            "[ZSTR teardown] Channel-B fsync(fd={}) returned {} \
                             (errno={})",
                            ghost_fd,
                            rc_b,
                            *libc::__errno_location()
                        );
                    } else {
                        log::info!(
                            "[ZSTR teardown] ✓ Channel-B fsync(fd={}) committed",
                            ghost_fd
                        );
                    }
                }
            }

            // ── WP4 conservation close-out.  `expected` / `produced` /
            //    `backpressure_events` come from the per-stream pinned control
            //    block the device slot-authority kernel maintains; the host
            //    never guesses them.
            let expected = ring.device_epochs();
            let produced = ring.device_produced();
            let backpressure_events = ring.device_backpressure_events();
            let backpressure_error = ring.backpressure_error();
            let backpressure_error_slot = ring.backpressure_error_slot();

            if terminal_error.is_none() && produced != persisted {
                terminal_error = Some(format!(
                    "ZSTR_CONSERVATION_BREAK: device produced {} records, {} persisted",
                    produced, persisted
                ));
            }
            if terminal_error.is_none() && expected != produced + backpressure_events {
                terminal_error = Some(format!(
                    "ZSTR_EPOCH_ACCOUNTING_BREAK: expected={} produced={} \
                     backpressure_events={}",
                    expected, produced, backpressure_events
                ));
            }

            let frames_dropped = produced.saturating_sub(persisted);

            log::info!(
                "[ZSTR consumer] exiting. expected={} produced={} persisted={} \
                 duplicates={} gaps={} backpressure_events={} bp_error={} \
                 ring_high_water={} g29_nonfinite={} bytes={} MB fsync_ok={} \
                 terminal_error={:?} path={:?}",
                expected,
                produced,
                persisted,
                duplicates,
                gaps,
                backpressure_events,
                backpressure_error,
                ring_high_water,
                g29_nonfinite_events,
                bytes_written >> 20,
                fsync_ok,
                terminal_error,
                output_path
            );

            ZstrStats {
                frames_written: persisted,
                frames_dropped,
                bytes_written,
                expected,
                produced,
                persisted,
                duplicates,
                gaps,
                expected_step_cadence,
                cadence_mismatches,
                first_device_step,
                last_device_step,
                ring_high_water,
                backpressure_events,
                backpressure_error,
                backpressure_error_slot,
                g29_nonfinite_events,
                g29_first_device_step,
                observed_schema_version,
                frame_size: frame_size as u64,
                n_atoms: ring.n_atoms as u64,
                fsync_ok,
                output_path: stats_path,
                terminal_error,
            }
        })
        .expect("Failed to spawn ZSTR consumer thread")
}

// ─── ZSTR C-ABI Launcher FFI ────────────────────────────────────────────────
//
// C-ABI wrappers compiled in libzstr_kernels.a.  Called during the
// cuStreamBeginCapture window on `telemetry_stream`; each call records a
// kernel node into the in-progress CUgraph.
//
// `zstr_launch_pos_stage`  — launches zstr_pos_stage_f4_kernel (n_atoms × 3
//   floats, VRAM → pinned slot, vectorised float4 STG.NC).
// `zstr_launch_fence_signal` — launches zstr_signal_completion_kernel (1×1
//   thread; __threadfence_system() + atomicExch(fence, 1)).

pub mod ffi {
    use std::ffi::c_void;

    extern "C" {
        /// WP4 per-stream-slot launcher: kernel reads `*d_active_slot` at
        /// execution time. `base_pinned` is slot-0 header start;
        /// `inter_slot_stride` is bytes between slot bases (== frame_size);
        /// `pos_offset_in_slot` is the header-size offset to positions payload.
        pub fn zstr_launch_pos_stage(
            base_pinned: *mut c_void,
            inter_slot_stride: u32,
            pos_offset_in_slot: u32,
            src_vram: *const c_void,
            n_atoms: u32,
            d_active_slot: *const c_void,
            stream: *mut c_void,
        ) -> i32;

        /// WP4 per-stream-slot launcher: fence signal. `base_fence` is slot-0
        /// completion_fence; kernel rolls slot via `*d_active_slot` and raises
        /// no fence at all when the epoch never claimed a slot.
        pub fn zstr_launch_fence_signal(
            base_fence: *mut c_void,
            inter_slot_stride: u32,
            d_active_slot: *const c_void,
            stream: *mut c_void,
        ) -> i32;

        /// DEPRECATED (WP4), preserved and reachable: stream-ordered device
        /// write of an explicit slot index into a PER-STREAM `d_active_slot`
        /// cell.
        ///
        /// The pre-WP4 form took only `(slot, stream)` because the slot lived
        /// in a module-scope `__device__` symbol; that symbol was ONE object
        /// shared by all 8 streams in the single CUDA context/module, so a
        /// slot written for stream A could be consumed by stream B's kernels.
        /// The symbol is deleted; this is the per-stream replacement.
        ///
        /// NOT used by the production path: only
        /// `prism_zstr_device_slot_update_launch` also stamps the frame
        /// header, and an unstamped claim reproduces the unparseable-file
        /// defect this work package exists to fix.
        pub fn prism_zstr_set_active_slot_ptr(
            d_active_slot: *mut c_void,
            slot: u32,
            stream: *mut c_void,
        ) -> i32;

        /// **WP4 §A — ZSTR slot authority.** Single-thread kernel recorded
        /// FIRST in each captured graph-body iteration. Three jobs:
        ///
        /// 1. Bounded backpressure on the target slot's `completion_fence`.
        ///    On expiry it records `ZSTR_BP_ERR_SPIN_EXPIRED` in
        ///    `d_backpressure_error`, writes `ZSTR_SLOT_NONE` into
        ///    `d_active_slot`, and returns WITHOUT claiming — an unpersisted
        ///    record is never overwritten and the wait never hangs.
        /// 2. Claims `(*d_step_counter) % n_slots` into the per-stream
        ///    `d_active_slot`.
        /// 3. Stamps the frozen 4096-byte header for that slot: `frame_idx`
        ///    (offset 0, u64) = the authoritative device step, `dt` (8, f32,
        ///    PICOSECONDS), `adjudication_code` (12, u32, zeroed here and
        ///    re-stamped by the audit kernel), `n_atoms` (20, u32), `gear_id`
        ///    (24, u32), `stream_id` (48, u32), `schema_version` (52, u32).
        ///    `completion_fence` / `force_norm` / `external_work` /
        ///    `potential_energy` are NOT touched.
        ///
        /// Before WP4 no kernel wrote the header at all: the offline parser
        /// `bin/prism_ghost_evidence_scan.rs` derives its file stride from
        /// `n_atoms` at offset 20, read 0, and abandoned the file with
        /// `parse_status="first_header_only"`.
        ///
        /// `d_backpressure_error` points at the per-stream status block
        /// (`ZSTR_BP_OFF_*`); may be null. `backpressure_spin_limit` of 0
        /// disables the bounded wait.
        ///
        /// Returns 0 on success, non-zero `cudaError_t` on launch failure.
        pub fn prism_zstr_device_slot_update_launch(
            d_step_counter: *const c_void, // u64 device pointer
            n_slots: u32,
            base_pinned: *mut c_void,
            inter_slot_stride: u32,
            d_active_slot: *mut c_void, // u32* per-stream
            n_atoms: u32,
            dt: f32,
            gear_id: u32,
            stream_id: u32,
            schema_version: u32,
            backpressure_spin_limit: u32,
            d_backpressure_error: *mut c_void,
            stream: *mut c_void,
        ) -> i32;

        /// T11 — Action-Recovery force exfiltration (DMA + Σ‖F‖² atomic-add).
        ///
        /// Records ONE kernel node onto `stream`:
        /// `zstr_force_stage_f4_kernel` — float4 LDG.E.128 from `d_forces` →
        /// STG.E.128 to the pinned slot's force payload, plus per-warp
        /// `__shfl_down_sync` butterfly reduce of Σ Fᵢ² with the warp leader
        /// `atomicAdd`-ing directly into the active slot's `force_norm`
        /// field (offset 28). The field holds the running sum-of-squares
        /// at this point; `zstr_launch_force_norm_sqrt` finalises it.
        ///
        /// `base_pinned`              : slot-0 header start.
        /// `inter_slot_stride`        : `ZstrRing::frame_size`.
        /// `force_offset_in_slot`     : forces payload offset
        ///                              (== `ZstrRing::forces_offset_in_slot()`).
        /// `force_norm_offset_in_slot`: 28 (offset_of_force_norm).
        /// `src_d_forces`             : `d_forces` device pointer
        ///                              (NhsAmberFusedEngine).
        /// `n_atoms`                  : atom count.
        /// `d_active_slot`            : per-stream slot cell
        ///                              (`ZstrRing::active_slot_ptr()`).
        ///
        /// Caller contract: the pinned `force_norm` field MUST be 0.0f
        /// before this kernel atomic-adds. The Reaper resets it alongside
        /// `completion_fence` once it has read the slot; the initial state
        /// is zero from `ZstrRing::allocate`.
        pub fn zstr_launch_force_stage(
            base_pinned: *mut c_void,
            inter_slot_stride: u32,
            force_offset_in_slot: u32,
            force_norm_offset_in_slot: u32,
            src_d_forces: *const c_void,
            n_atoms: u32,
            d_active_slot: *const c_void,
            stream: *mut c_void,
        ) -> i32;

        /// T11 — Single-thread post-pass: in-place sqrtf of the active
        /// slot's `force_norm`, converting the accumulated Σ‖F‖² to
        /// ‖F‖₂ before the consumer reads it.  NaN propagates verbatim
        /// (host-side G29 Reaper traps non-finite values).
        pub fn zstr_launch_force_norm_sqrt(
            base_pinned: *mut c_void,
            inter_slot_stride: u32,
            force_norm_offset_in_slot: u32,
            d_active_slot: *const c_void,
            stream: *mut c_void,
            d_force_norm_mirror: *mut c_void,
        ) -> i32;

        /// **M1.2.18.5 / WP4** — single-thread audit-field stage. Snapshots
        /// `W_ext` and `V_t` into the active slot and, WP4, stamps
        /// `adjudication_code` (slot offset 12) from `adj + 52`. This is the
        /// only ZSTR node captured downstream of the Adjudicator, so it is
        /// the sole owner of that header field.
        pub fn zstr_launch_stage_audit(
            base_pinned: *mut c_void,
            inter_slot_stride: u32,
            external_work_offset_in_slot: u32,
            potential_energy_offset_in_slot: u32,
            adjudication_code_offset_in_slot: u32,
            adj: *const c_void,
            d_active_slot: *const c_void,
            stream: *mut c_void,
        ) -> i32;
    }
}
