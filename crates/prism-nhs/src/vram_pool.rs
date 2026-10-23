//! F2 — Stream-Ordered Memory Pool + VRAM Audit telemetry.
//!
//! Per Blackwell Convergence mandate §3 (operator directive
//! 2026-04-29). Replaces static "worst-case" pre-allocation with a
//! `cudaMemPool`-backed stream-ordered allocator (`cudaMallocFromPoolAsync`
//! / `cudaFreeAsync`) and a global VRAM telemetry struct that the F1
//! Adjudicator's SWITCH node consumes as the Case-2 Violation trigger.
//!
//! # Caller contract
//!
//! 1. Construct one [`VramPool`] per device. The Drop impl returns
//!    the pool handle to the driver via `cudaMemPoolDestroy`. With
//!    `ReleaseThreshold = UINT64_MAX` (set at create time) the
//!    driver does NOT return pool memory to the OS while the pool
//!    is alive — every captured-graph replay sees a stable mempool.
//! 2. Construct one [`VramAuditDevice`] per stream/replica. The
//!    Drop impl frees the device-side audit struct.
//! 3. For every allocation:
//!      a. `pool.alloc_async(size, stream)` → device pointer.
//!      b. IMMEDIATELY: `audit.record_alloc(size, budget, stream)`.
//!    The two kernel launches are stream-ordered, so the telemetry
//!    update is guaranteed to retire after the allocation node in
//!    any captured graph.
//! 4. For every free:
//!      a. `pool.free_async(ptr, stream)`.
//!      b. `audit.record_free(size, stream)`.
//! 5. The host reads the audit via [`VramAuditDevice::snapshot`]
//!    after `stream.synchronize()`. The F1 SWITCH node will read it
//!    via a captured-graph conditional-node update path (M1.2.5).
//!
//! # Concurrency model
//!
//! Every telemetry kernel runs as a single-thread launch
//! (`<<<1, 1, 0, stream>>>`). Single-threaded launches keep the
//! semantics race-free under concurrent invocations from different
//! streams: every atomic op on the audit struct is a single-
//! instruction memory transaction, so cross-stream concurrent
//! updates compose deterministically.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ============================================================================
// VramAudit POD struct (FFI-stable)
// ============================================================================

/// Mirror of the C-side `VramAudit` in `vram_pool.cuh`. Layout-pinned
/// at 24 bytes, 8-byte aligned. The trailing 4-byte padding (after
/// the u32 `pool_exhaustion_flag`) is implicit per Rust + C ABI rules
/// — both sides round size up to align-of-largest-field (8 B).
///
/// Three fields:
///
/// * [`current_allocated_bytes`] — sum of every `alloc_size` recorded
///   minus every `free_size`. The "live" working-set count.
/// * [`peak_high_water_mark`] — max value `current_allocated_bytes`
///   ever reached during the audit's lifetime. Monotonic; never
///   decrements on free.
/// * [`pool_exhaustion_flag`] — 0 = OK, 1 = ABORT. Set sticky-
///   irreversibly when any `record_alloc` would push
///   `current_allocated_bytes` above the supplied `budget`.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VramAudit {
    pub current_allocated_bytes: u64,
    pub peak_high_water_mark: u64,
    pub pool_exhaustion_flag: u32,
    /// Trailing padding to align the next array element to 8 bytes.
    /// Initialized to zero by the C-side `prism_vram_audit_zero_kernel`
    /// (which writes the whole struct, padding included, via
    /// individual field assignments) and never read.
    _pad: u32,
}

impl VramAudit {
    /// All-zeros initial state. Equivalent to `audit_init`'s effect
    /// on the device-side struct. Used by tests + as a sane default.
    pub const fn zero() -> Self {
        Self {
            current_allocated_bytes: 0,
            peak_high_water_mark: 0,
            pool_exhaustion_flag: 0,
            _pad: 0,
        }
    }

    /// Sentinel for `pool_exhaustion_flag`: pool budget intact.
    pub const FLAG_OK: u32 = 0;
    /// Sentinel for `pool_exhaustion_flag`: budget exceeded; routes
    /// the F1 SWITCH node to Case-2 Violation.
    pub const FLAG_VIOLATION: u32 = 1;

    /// True iff `pool_exhaustion_flag == FLAG_VIOLATION`.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.pool_exhaustion_flag == Self::FLAG_VIOLATION
    }
}

impl Default for VramAudit {
    fn default() -> Self {
        Self::zero()
    }
}

/// Native pool attributes. Pool-local accounting, sampled sequentially without
/// synchronization or resetting the lifetime high-water marks. A caller needing
/// retired allocation/free state must synchronize its owners before this query.
/// This is not a complete engine footprint: cudarc, graphs, modules and other
/// pools are outside this pool's accounting domain.
#[repr(C, align(8))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VramPoolMemorySnapshot {
    pub used_current_bytes: u64,
    pub used_high_bytes: u64,
    pub reserved_current_bytes: u64,
    pub reserved_high_bytes: u64,
}

// ============================================================================
// FFI surface
// ============================================================================

#[cfg(feature = "gpu")]
#[allow(dead_code)]
mod ffi {
    use super::{VramAudit, VramPoolMemorySnapshot};

    pub type CudaError = i32;
    pub const CUDA_SUCCESS: CudaError = 0;

    extern "C" {
        /// Sentinel `0xF2A4`. Confirms the static archive linked
        /// correctly and the FFI ABI is round-tripping.
        pub fn prism_vram_pool_link_probe() -> u32;

        pub fn prism_vram_pool_create(
            device_id: i32,
            out_pool: *mut *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_vram_pool_destroy(pool: *mut std::ffi::c_void) -> CudaError;

        pub fn prism_vram_pool_alloc_async(
            pool: *mut std::ffi::c_void,
            size: u64,
            stream: *mut std::ffi::c_void,
            out_ptr: *mut *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_vram_pool_memory_snapshot(
            pool: *mut std::ffi::c_void,
            out_snapshot: *mut VramPoolMemorySnapshot,
        ) -> CudaError;

        pub fn prism_vram_pool_free_async(
            ptr: *mut std::ffi::c_void,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_vram_audit_init(
            d_audit: *mut VramAudit,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_vram_audit_record_alloc(
            d_audit: *mut VramAudit,
            alloc_size: u64,
            budget: u64,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_vram_audit_record_free(
            d_audit: *mut VramAudit,
            free_size: u64,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;
    }
}

/// Safe wrapper over the FFI link-probe. Returns `0xF2A4`.
#[cfg(feature = "gpu")]
pub fn link_probe() -> u32 {
    unsafe { ffi::prism_vram_pool_link_probe() }
}

// ============================================================================
// VramPool — RAII wrapper around `cudaMemPool_t`
// ============================================================================

/// RAII handle for a stream-ordered `cudaMemPool`.
///
/// On construction a fresh pool is created on the requested device
/// with the three Blackwell-required attributes:
///   - `cudaMemPoolAttrReleaseThreshold` = UINT64_MAX (driver never
///     returns pool memory to the OS during the pool's lifetime —
///     load-bearing for captured-graph stability).
///   - `cudaMemPoolAttrReuseAllowOpportunistic` = false.
///   - `cudaMemPoolAttrReuseAllowInternalDependencies` = false.
///
/// On Drop the pool is destroyed via `cudaMemPoolDestroy`. If a Drop
/// returns an error it is logged via `eprintln!` rather than panicking;
/// CUDA pool-destroy errors at process shutdown are typically benign
/// (e.g., already-shutdown context) and panicking here would
/// double-fault the program.
#[cfg(feature = "gpu")]
pub struct VramPool {
    handle: *mut std::ffi::c_void,
    device_id: i32,
}

#[cfg(feature = "gpu")]
unsafe impl Send for VramPool {}
// VramPool is NOT Sync: the underlying cudaMemPool_t is documented
// as thread-safe for alloc/free calls but the wrapper's Drop
// implementation forbids concurrent destroy attempts. Restrict to
// single-thread or `Arc<Mutex<VramPool>>` if cross-thread use is
// needed.

#[cfg(feature = "gpu")]
impl VramPool {
    /// Create a new pool on `device_id`. Returns the wrapper or a
    /// formatted error string.
    pub fn new(device_id: i32) -> Result<Self, String> {
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe { ffi::prism_vram_pool_create(device_id, &mut handle) };
        if rc != ffi::CUDA_SUCCESS {
            return Err(format!(
                "prism_vram_pool_create failed: cudaError {} on device {}",
                rc, device_id
            ));
        }
        if handle.is_null() {
            return Err(format!(
                "prism_vram_pool_create returned cudaSuccess but null handle on device {}",
                device_id
            ));
        }
        Ok(Self { handle, device_id })
    }

    /// Raw pool handle as `usize`. Threaded into FFI calls that need
    /// the underlying `cudaMemPool_t`.
    #[inline]
    pub fn raw_handle(&self) -> usize {
        self.handle as usize
    }

    /// Device id this pool was created on.
    #[inline]
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Allocate `size` bytes from this pool on `stream`. Returns the
    /// raw device pointer as `usize`. Caller is responsible for
    /// pairing this with [`Self::free_async`] on a stream where the
    /// last user of the pointer has retired (stream-ordered free
    /// semantics — see CUDA Mempool docs).
    pub fn alloc_async(&self, size: u64, stream_handle: usize) -> Result<usize, String> {
        let mut out: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe {
            ffi::prism_vram_pool_alloc_async(
                self.handle,
                size,
                stream_handle as *mut std::ffi::c_void,
                &mut out,
            )
        };
        if rc != ffi::CUDA_SUCCESS {
            return Err(format!(
                "prism_vram_pool_alloc_async({} bytes) failed: cudaError {}",
                size, rc
            ));
        }
        if out.is_null() {
            return Err(format!(
                "prism_vram_pool_alloc_async({} bytes) returned cudaSuccess but null pointer",
                size
            ));
        }
        // Operator Amendment 3.9 §4 — F2 Pool 256-byte alignment guard.
        //
        // cudaMallocFromPoolAsync (the C++-side backing allocator) is
        // documented to return at least 256-byte aligned addresses.
        // Blackwell sm_120 LDG.E.128 / RED.E.ADD.V4 vector ops trap on
        // any sub-16-byte misalignment; cache-sector-aligned (256 B)
        // exceeds that floor.  This assertion catches any future
        // allocator regression (e.g., a sub-allocator that subdivides
        // a parent block on non-256-byte boundaries) at the point of
        // mint, before the pointer enters the captured graph and the
        // hardware traps deep inside a kernel launch.
        let ptr = out as usize;
        assert!(
            ptr % 256 == 0,
            "F2 POOL ALIGNMENT VIOLATION: alloc_async({} bytes) returned \
             0x{:x} which is not 256-byte aligned (mod 256 = {}). \
             Blackwell sm_120 vector loads will trap. HALT.",
            size,
            ptr,
            ptr % 256
        );
        Ok(ptr)
    }

    /// Read actual CUDA pool attributes. Does not synchronize, trim, reset peaks
    /// or change any reuse policy. Unavailable readings return an error.
    pub fn memory_snapshot(&self) -> Result<VramPoolMemorySnapshot, String> {
        let mut sampled = VramPoolMemorySnapshot::default();
        let rc = unsafe { ffi::prism_vram_pool_memory_snapshot(self.handle, &mut sampled) };
        if rc != ffi::CUDA_SUCCESS {
            return Err(format!("prism_vram_pool_memory_snapshot failed: cudaError {}", rc));
        }
        Ok(sampled)
    }

    /// Free a pointer back to this pool on `stream`. Stream-ordered.
    pub fn free_async(&self, device_ptr: usize, stream_handle: usize) -> Result<(), String> {
        let rc = unsafe {
            ffi::prism_vram_pool_free_async(
                device_ptr as *mut std::ffi::c_void,
                stream_handle as *mut std::ffi::c_void,
            )
        };
        if rc != ffi::CUDA_SUCCESS {
            return Err(format!(
                "prism_vram_pool_free_async failed: cudaError {}",
                rc
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "gpu")]
impl Drop for VramPool {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        let rc = unsafe { ffi::prism_vram_pool_destroy(self.handle) };
        if rc != ffi::CUDA_SUCCESS {
            // Process-shutdown / context-already-destroyed errors
            // here are benign; panicking would mask the real cause
            // of an outer failure. Log and continue.
            eprintln!(
                "[VramPool::drop] prism_vram_pool_destroy returned cudaError {}",
                rc
            );
        }
        self.handle = std::ptr::null_mut();
    }
}

// ============================================================================
// VramAuditDevice — RAII wrapper around the device-side audit struct
// ============================================================================

/// Owns one device-side [`VramAudit`] struct (24 bytes) and the
/// stream telemetry kernels are launched on. The host reads via
/// [`Self::snapshot`] after `stream.synchronize()`.
#[cfg(feature = "gpu")]
pub struct VramAuditDevice {
    /// 24 bytes of device memory holding the [`VramAudit`] struct.
    /// Owned by this wrapper; freed on Drop.
    d_audit: cudarc::driver::CudaSlice<u8>,
    /// Stream the telemetry kernels are launched on. Held as a
    /// reference so the wrapper can launch additional updates later.
    stream: Arc<cudarc::driver::CudaStream>,
}

#[cfg(feature = "gpu")]
impl VramAuditDevice {
    /// Allocate the device-side audit struct on `stream` and zero
    /// every field. The zero pass is enqueued onto the same stream
    /// so it's stream-ordered with subsequent telemetry updates.
    pub fn new(stream: Arc<cudarc::driver::CudaStream>) -> Result<Self, String> {
        let d_audit = stream
            .alloc_zeros::<u8>(std::mem::size_of::<VramAudit>())
            .map_err(|e| format!("alloc d_audit: {:?}", e))?;
        // Explicit zero kernel — the alloc_zeros above already
        // zeroes the bytes via cudaMemset, but the explicit kernel
        // makes the "init" semantics observable on a stream and
        // ensures any future change to the audit struct that adds
        // padding-sensitive fields is exercised by the same code path.
        //
        // Scoped block: the cudarc `device_ptr` borrow guard
        // (`_g`) must drop before we move `d_audit` and `stream`
        // into `Self` at the end of this fn.
        {
            use cudarc::driver::DevicePtr;
            let (audit_dev, _g) = d_audit.device_ptr(&stream);
            let raw_stream = stream.cu_stream() as usize;
            let rc = unsafe {
                ffi::prism_vram_audit_init(
                    audit_dev as *mut VramAudit,
                    raw_stream as *mut std::ffi::c_void,
                )
            };
            if rc != ffi::CUDA_SUCCESS {
                return Err(format!("prism_vram_audit_init failed: cudaError {}", rc));
            }
        }
        Ok(Self { d_audit, stream })
    }

    /// Record a single allocation. Idempotent w.r.t. the
    /// pool_exhaustion_flag (atomicCAS sets it 0→1 once; subsequent
    /// allocations over budget are no-ops on the flag).
    pub fn record_alloc(&self, alloc_size: u64, budget: u64) -> Result<(), String> {
        use cudarc::driver::DevicePtr;
        let (audit_dev, _g) = self.d_audit.device_ptr(&self.stream);
        let raw_stream = self.stream.cu_stream() as usize;
        let rc = unsafe {
            ffi::prism_vram_audit_record_alloc(
                audit_dev as *mut VramAudit,
                alloc_size,
                budget,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        if rc != ffi::CUDA_SUCCESS {
            return Err(format!(
                "prism_vram_audit_record_alloc failed: cudaError {}",
                rc
            ));
        }
        Ok(())
    }

    /// Record a single free. Subtracts `free_size` from
    /// `current_allocated_bytes`; does not touch `peak_high_water_mark`
    /// or `pool_exhaustion_flag` (those are sticky high-water signals).
    pub fn record_free(&self, free_size: u64) -> Result<(), String> {
        use cudarc::driver::DevicePtr;
        let (audit_dev, _g) = self.d_audit.device_ptr(&self.stream);
        let raw_stream = self.stream.cu_stream() as usize;
        let rc = unsafe {
            ffi::prism_vram_audit_record_free(
                audit_dev as *mut VramAudit,
                free_size,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        if rc != ffi::CUDA_SUCCESS {
            return Err(format!(
                "prism_vram_audit_record_free failed: cudaError {}",
                rc
            ));
        }
        Ok(())
    }

    /// Synchronize the stream, then dtoh-copy the device-side audit
    /// struct into a host-side [`VramAudit`]. Called by the F1
    /// Adjudicator's host-side decision point AND by tests.
    pub fn snapshot(&self) -> Result<VramAudit, String> {
        self.stream
            .synchronize()
            .map_err(|e| format!("stream sync: {:?}", e))?;
        let mut bytes = vec![0u8; std::mem::size_of::<VramAudit>()];
        self.stream
            .memcpy_dtoh(&self.d_audit, &mut bytes)
            .map_err(|e| format!("dtoh audit: {:?}", e))?;
        // SAFETY: we just dtoh'd exactly size_of::<VramAudit>() bytes
        // into a properly-aligned-via-Vec<u8> buffer, and VramAudit is
        // #[repr(C, align(8))] with no internal references. The
        // alignment guarantee from Vec<u8> is 1 byte; we copy through
        // an unaligned read via std::ptr::read_unaligned.
        let audit = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const VramAudit) };
        Ok(audit)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_memory_snapshot_layout_is_32_bytes_8_aligned() {
        assert_eq!(std::mem::size_of::<VramPoolMemorySnapshot>(), 32);
        assert_eq!(std::mem::align_of::<VramPoolMemorySnapshot>(), 8);
    }

    #[cfg(feature = "gpu")]
    #[ignore = "GPU_RUNTIME_REQUIRED: reads real pool attributes after actual alloc/free"]
    #[test]
    fn pool_memory_snapshot_tracks_live_bytes_without_resetting_peak() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA required");
        let stream = ctx.new_stream().expect("stream");
        let pool = VramPool::new(0).expect("pool");
        let other_pool = VramPool::new(0).expect("independent pool");
        let initial = pool.memory_snapshot().expect("initial snapshot");
        assert_eq!(initial.used_current_bytes, 0);
        let ptr = pool.alloc_async(8 * 1024 * 1024, stream.cu_stream() as usize).expect("allocate");
        stream.synchronize().expect("retire allocation");
        let allocated = pool.memory_snapshot().expect("allocated snapshot");
        assert!(allocated.used_current_bytes >= 8 * 1024 * 1024);
        assert!(allocated.used_high_bytes >= allocated.used_current_bytes);
        assert!(allocated.reserved_current_bytes >= allocated.used_current_bytes);
        assert_eq!(other_pool.memory_snapshot().expect("other pool").used_current_bytes, 0);
        pool.free_async(ptr, stream.cu_stream() as usize).expect("free");
        stream.synchronize().expect("retire free");
        let freed = pool.memory_snapshot().expect("freed snapshot");
        assert_eq!(freed.used_current_bytes, 0);
        assert_eq!(freed.used_high_bytes, allocated.used_high_bytes);
        assert_eq!(freed.reserved_high_bytes, allocated.reserved_high_bytes);
        assert_eq!(pool.memory_snapshot().expect("repeat snapshot"), freed);
        println!("{}", serde_json::json!({"before": initial, "allocated": allocated, "freed": freed}));
    }

    #[test]
    fn vram_audit_layout_is_24_bytes_8_aligned() {
        // The C-side static_assert pins sizeof(VramAudit) == 24 and
        // alignof == 8. The Rust mirror MUST match.
        assert_eq!(std::mem::size_of::<VramAudit>(), 24);
        assert_eq!(std::mem::align_of::<VramAudit>(), 8);
    }

    #[test]
    fn vram_audit_zero_default_state() {
        let a = VramAudit::zero();
        assert_eq!(a.current_allocated_bytes, 0);
        assert_eq!(a.peak_high_water_mark, 0);
        assert_eq!(a.pool_exhaustion_flag, VramAudit::FLAG_OK);
        assert!(!a.is_exhausted());
    }

    #[test]
    fn vram_audit_is_exhausted_returns_true_when_flag_set() {
        let mut a = VramAudit::zero();
        a.pool_exhaustion_flag = VramAudit::FLAG_VIOLATION;
        assert!(a.is_exhausted());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn link_probe_returns_sentinel() {
        // Confirms vram_pool.cu linked into the static archive and
        // the FFI ABI is round-tripping. Sentinel pinned at 0xF2A4.
        assert_eq!(super::link_probe(), 0x0000_F2A4);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn pool_create_destroy_round_trip() {
        use cudarc::driver::CudaContext;
        let _ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[vram_pool] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        // Create + Drop. The Drop impl calls destroy. If destroy
        // fails the test prints an eprintln but does not panic;
        // we assert only on the create path.
        let pool = VramPool::new(0).expect("pool create");
        assert_ne!(pool.raw_handle(), 0, "pool handle must be non-null");
        assert_eq!(pool.device_id(), 0);
        // Implicit drop here via end of scope — exercises destroy.
    }

    #[cfg(feature = "gpu")]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn pool_alloc_free_round_trip() {
        use cudarc::driver::CudaContext;
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[vram_pool alloc] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream create");

        let pool = VramPool::new(0).expect("pool create");
        let stream_handle = stream.cu_stream() as usize;

        // Allocate 4 KiB — small enough not to OOM any device.
        let dev_ptr = pool.alloc_async(4096, stream_handle).expect("alloc 4 KiB");
        assert_ne!(dev_ptr, 0, "device pointer must be non-null");

        // Free + sync. The free is stream-ordered: returns
        // immediately, actual reclaim happens after stream catches up.
        pool.free_async(dev_ptr, stream_handle).expect("free");
        stream.synchronize().expect("stream sync");
    }

    #[cfg(feature = "gpu")]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn audit_telemetry_high_water_and_exhaustion_flag() {
        // ── F2 telemetry struct verification (operator's mandate ──
        // §3.2 audit gate).
        //
        // Scenario: budget = 5000 bytes. Record 5 allocations of
        // 1000 bytes each → current = 5000 (boundary, NOT
        // exhausted). Record 1 more (alloc_size = 1000) → current =
        // 6000 > budget, exhaustion flag set. peak = 6000.
        // Record 2 frees of 1000 each → current = 4000.
        // peak_high_water_mark stays at 6000 (sticky monotonic).
        // pool_exhaustion_flag stays at 1 (sticky).
        //
        // This exercises every field of the audit struct and every
        // telemetry kernel exactly once.

        use cudarc::driver::CudaContext;
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "[vram_audit telemetry] CUDA unavailable: {:?} — skipping",
                    e
                );
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream create");

        let audit = VramAuditDevice::new(stream.clone()).expect("audit new");

        // Initial snapshot: all zeros.
        let s0 = audit.snapshot().expect("snapshot 0");
        assert_eq!(
            s0,
            VramAudit::zero(),
            "audit init kernel did not zero every field"
        );

        // 5 record_alloc of 1000 bytes each, budget = 5000.
        let budget: u64 = 5000;
        for _ in 0..5 {
            audit.record_alloc(1000, budget).expect("record_alloc");
        }
        let s1 = audit.snapshot().expect("snapshot 1");
        assert_eq!(s1.current_allocated_bytes, 5000);
        assert_eq!(s1.peak_high_water_mark, 5000);
        assert_eq!(
            s1.pool_exhaustion_flag,
            VramAudit::FLAG_OK,
            "5000 bytes at boundary 5000 must NOT trip exhaustion (>, not >=)"
        );

        // 1 more record_alloc — pushes over budget.
        audit
            .record_alloc(1000, budget)
            .expect("record_alloc over budget");
        let s2 = audit.snapshot().expect("snapshot 2");
        assert_eq!(s2.current_allocated_bytes, 6000);
        assert_eq!(s2.peak_high_water_mark, 6000);
        assert_eq!(
            s2.pool_exhaustion_flag,
            VramAudit::FLAG_VIOLATION,
            "6000 > budget 5000 must trip exhaustion flag"
        );
        assert!(s2.is_exhausted());

        // 2 record_free of 1000 each — current drops, peak stays,
        // flag stays.
        audit.record_free(1000).expect("record_free 1");
        audit.record_free(1000).expect("record_free 2");
        let s3 = audit.snapshot().expect("snapshot 3");
        assert_eq!(
            s3.current_allocated_bytes, 4000,
            "current_allocated_bytes must decrement on record_free"
        );
        assert_eq!(
            s3.peak_high_water_mark, 6000,
            "peak_high_water_mark must NOT decrement on free (monotonic)"
        );
        assert_eq!(
            s3.pool_exhaustion_flag,
            VramAudit::FLAG_VIOLATION,
            "exhaustion flag must stay sticky after a free recovers headroom"
        );
    }

    #[cfg(feature = "gpu")]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn pool_plus_audit_end_to_end() {
        // Full F2 path: pool alloc → telemetry record → pool free
        // → telemetry record. Pool budget intentionally LARGER than
        // the audit budget so the pool succeeds while the audit
        // detects the budget violation. The audit is the SOURCE OF
        // TRUTH for exhaustion, not the pool's own success/fail.
        use cudarc::driver::CudaContext;
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[vram pool+audit e2e] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream create");
        let pool = VramPool::new(0).expect("pool create");
        let audit = VramAuditDevice::new(stream.clone()).expect("audit new");
        let stream_handle = stream.cu_stream() as usize;

        let small_budget: u64 = 4096; // 4 KiB
        let alloc_size: u64 = 8192; // 8 KiB — over budget intentionally

        // Pool alloc succeeds (the device has plenty of memory) but
        // the audit budget is violated.
        let dev_ptr = pool
            .alloc_async(alloc_size, stream_handle)
            .expect("pool alloc");
        audit
            .record_alloc(alloc_size, small_budget)
            .expect("audit record_alloc");

        let s = audit.snapshot().expect("snapshot");
        assert_eq!(s.current_allocated_bytes, alloc_size);
        assert_eq!(s.peak_high_water_mark, alloc_size);
        assert!(
            s.is_exhausted(),
            "single 8 KiB alloc against 4 KiB budget must trip the flag"
        );

        // Cleanup.
        pool.free_async(dev_ptr, stream_handle).expect("pool free");
        audit.record_free(alloc_size).expect("audit record_free");
        stream.synchronize().expect("stream sync");
    }
}

/// Sparse construction/capture telemetry shared by standard and coupled callers.
/// Device values are global; a supplied pool reading remains pool-owner scoped.
/// This adds no explicit CUDA synchronization or allocator-policy mutation.
#[cfg(feature = "gpu")]
pub fn record_allocation_boundary(
    writer: &mut Option<std::io::BufWriter<std::fs::File>>,
    context: &cudarc::driver::CudaContext,
    source_stream_id: usize,
    boundary: &str,
    pool: Option<std::result::Result<serde_json::Value, String>>,
) -> anyhow::Result<()> {
    use std::io::Write;
    let Some(writer) = writer.as_mut() else { return Ok(()); };
    let query_begin = std::time::Instant::now();
    let timestamp_unix_ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    // bind_to_thread checks cudarc's deferred error channel. Propagate it;
    // telemetry must not consume a physics error and allow execution to proceed.
    context.bind_to_thread().map_err(|e| anyhow::anyhow!("P1 allocation context bind: {:?}", e))?;
    let device_memory = match cudarc::driver::result::mem_get_info() {
        Ok((free, total)) => serde_json::json!({"status":"measured", "free_bytes":free, "total_bytes":total}),
        Err(error) => serde_json::json!({"status":"unavailable", "error":format!("{:?}",error)}),
    };
    let pipeline_pool = match pool {
        Some(Ok(value)) => serde_json::json!({"status":"measured", "measurement":value}),
        Some(Err(error)) => serde_json::json!({"status":"unavailable", "error":error}),
        None => serde_json::json!({"status":"not_attached_at_this_boundary"}),
    };
    serde_json::to_writer(&mut *writer, &serde_json::json!({
        "schema_kind":"prism.native_allocation_boundary.v1",
        "source_stream_id":source_stream_id, "boundary":boundary,
        "timestamp_unix_ns":timestamp_unix_ns,
        "device_id":context.cu_device(),
        "device_query_host_duration_ns":query_begin.elapsed().as_nanos(),
        "device_memory":device_memory,
        "device_memory_scope":"device_global_including_other_streams_contexts_and_processes",
        "device_global_delta_is_exclusive_stream_allocation":false,
        "pipeline_pool":pipeline_pool,
        "explicit_cuda_synchronization_added":false,
        "sampling":"sequential_host_queries_not_atomic_across_concurrent_workers",
        "allocation_owner_ledger":"not_available_for_all_engine_allocations"
    }))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

/// Construction-only receipt for one immutable allocation owner. This does not
/// attribute global VRAM changes to an engine or claim whole-ensemble capacity.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Serialize)]
pub struct ImmutableF32BankReceipt {
    pub scope: &'static str,
    pub content_sha256: String,
    pub elements: usize,
    pub allocation_bytes: usize,
    pub reused_existing_allocation: bool,
}

/// Share an exact initialized f32 snapshot only within the SAME Arc<CudaContext>.
/// No physics/detector state is modified. The weak cache cannot extend lifetime.
/// Every consumer must keep its Arc under its own stream/graph-retirement owner;
/// do not reuse a mutable buffer or transfer one consumer's retirement lease.
/// This setup-only path reads and hashes native bytes, then fences first
/// publication. It adds no query, copy, mutex or synchronization to MD replay.
#[cfg(feature = "gpu")]
pub fn intern_immutable_f32(
    stream: &Arc<cudarc::driver::CudaStream>,
    initialized: &cudarc::driver::CudaSlice<f32>,
) -> anyhow::Result<(Arc<cudarc::driver::CudaSlice<f32>>, ImmutableF32BankReceipt)> {
    use anyhow::Context;
    use sha2::Digest;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock, Weak};
    use cudarc::driver::CudaSlice;
    type Key = (usize, usize, [u8; 32]);
    type Banks = HashMap<Key, Weak<CudaSlice<f32>>>;
    static BANKS: OnceLock<Mutex<Banks>> = OnceLock::new();
    anyhow::ensure!(Arc::ptr_eq(stream.context(), initialized.context()),
        "immutable snapshot source and consumer have different context owners");
    anyhow::ensure!(!initialized.is_empty(), "immutable snapshot cannot be empty");
    let values = stream.clone_dtoh(initialized).context("read native immutable initializer")?;
    stream.synchronize().context("fence immutable initializer readback")?;
    // SAFETY: f32 has no padding, all elements are initialized by the checked
    // readback, and this immutable byte view cannot outlive `values`. Hash in
    // one contiguous pass; preserve signed zero and NaN payloads without a
    // second full-size host allocation or per-element hashing overhead.
    let bytes = unsafe { std::slice::from_raw_parts(
        values.as_ptr().cast::<u8>(), initialized.num_bytes(),
    ) };
    let digest: [u8;32] = sha2::Sha256::digest(bytes).into();
    let key = (Arc::as_ptr(stream.context()) as usize, initialized.len(), digest);
    let mut banks = BANKS.get_or_init(|| Mutex::new(HashMap::new())).lock()
        .map_err(|_| anyhow::anyhow!("immutable bank cache poisoned"))?;
    banks.retain(|_, bank| bank.strong_count() > 0);
    let (bank, reused) = if let Some(bank) = banks.get(&key).and_then(Weak::upgrade) {
        (bank, true)
    } else {
        let bank = stream.clone_dtod(initialized).context("allocate native immutable snapshot")?;
        // Publication happens before another engine can discover the pointer.
        // The owner keeps this stream/context alive until the last reader retires.
        stream.synchronize().context("publish native immutable snapshot")?;
        let bank = Arc::new(bank);
        banks.insert(key, Arc::downgrade(&bank));
        (bank, false)
    };
    Ok((bank, ImmutableF32BankReceipt {
        scope: "same_context_equal_initialized_bytes_immutable_owner",
        content_sha256: digest.iter().map(|byte| format!("{byte:02x}")).collect(),
        elements: initialized.len(),
        allocation_bytes: initialized.num_bytes(),
        reused_existing_allocation: reused,
    }))
}

#[cfg(all(test, feature = "gpu"))]
mod immutable_bank_gpu_tests {
    use super::*;
    use cudarc::driver::{CudaContext, DevicePtr};
    use crate::governor_device::CudaOwned;

    #[test]
    #[ignore = "GPU_RUNTIME_REQUIRED: actual immutable bank sharing and owner retirement"]
    fn immutable_bank_shares_equal_bytes_and_preserves_independent_owners() {
        let context = CudaContext::new(0).unwrap();
        let a = context.new_stream().unwrap();
        let b = context.new_stream().unwrap();
        // Exact bit comparison includes signed zero and a noncanonical NaN.
        let bits = [0.4f32.to_bits(), 0, 0x80000000, 0x7fc01234];
        let values: Vec<f32> = (0..4096).map(|i| f32::from_bits(bits[i%4])).collect();
        let mut source_a = a.clone_htod(&values).unwrap();
        let source_b = b.clone_htod(&values).unwrap();
        let (shared_a, first) = intern_immutable_f32(&a, &source_a).unwrap();
        let (shared_b, second) = intern_immutable_f32(&b, &source_b).unwrap();
        assert!(!first.reused_existing_allocation);
        assert!(second.reused_existing_allocation);
        assert_eq!(first.content_sha256, second.content_sha256);
        assert!(Arc::ptr_eq(&shared_a, &shared_b));
        assert_eq!(shared_a.device_ptr(&a).0, shared_b.device_ptr(&b).0);
        let weak = Arc::downgrade(&shared_a);
        let owner_a = CudaOwned::new(shared_a, a.clone());
        let owner_b = CudaOwned::new(shared_b, b.clone());
        a.memcpy_htod(&vec![9.0f32;values.len()], &mut source_a).unwrap();
        // Mutable threshold state changes do not change either immutable floor.
        let readback = b.clone_dtoh(&**owner_b).unwrap();
        assert!(readback.iter().zip(&values).all(|(x,y)| x.to_bits()==y.to_bits()));
        drop(owner_a);
        assert!(weak.upgrade().is_some());
        let readback = b.clone_dtoh(&**owner_b).unwrap();
        assert!(readback.iter().zip(&values).all(|(x,y)| x.to_bits()==y.to_bits()));
        drop(owner_b);
        assert!(weak.upgrade().is_none());
        let (_, after_retirement) = intern_immutable_f32(&b, &source_b).unwrap();
        assert!(!after_retirement.reused_existing_allocation);
    }

    #[test]
    #[ignore = "GPU_RUNTIME_REQUIRED: unequal values/context-owner isolation"]
    fn immutable_bank_rejects_cross_context_owner_and_never_shares_unequal_bytes() {
        let context = CudaContext::new(0).unwrap();
        let a = context.new_stream().unwrap();
        let other_context = CudaContext::new(0).unwrap();
        let b = other_context.new_stream().unwrap();
        // Cudarc new() retains the primary CUDA context. These are deliberately
        // distinct Arc<CudaContext> owners, not a claim of two physical devices.
        assert!(!Arc::ptr_eq(&context, &other_context));
        let source_a = a.clone_htod(&[0.4f32;64]).unwrap();
        let source_b = b.clone_htod(&[0.4f32;64]).unwrap();
        let changed = a.clone_htod(&[0.5f32;64]).unwrap();
        let (bank_a, _) = intern_immutable_f32(&a, &source_a).unwrap();
        let (bank_b, rb) = intern_immutable_f32(&b, &source_b).unwrap();
        let (bank_changed, rc) = intern_immutable_f32(&a, &changed).unwrap();
        assert!(!rb.reused_existing_allocation && !rc.reused_existing_allocation);
        assert!(!Arc::ptr_eq(&bank_a,&bank_b));
        assert!(!Arc::ptr_eq(&bank_a,&bank_changed));
        assert!(intern_immutable_f32(&b,&source_a).is_err());
        drop(CudaOwned::new(bank_a,a.clone()));
        drop(CudaOwned::new(bank_changed,a));
        drop(CudaOwned::new(bank_b,b));
    }
}

#[cfg(all(test,feature="gpu"))]
mod immutable_bank_native_engine_tests {
    use super::*;
    use cudarc::driver::{CudaContext,DevicePtr};
    use crate::governor_device::CudaOwned;
    #[test]
    #[ignore = "GPU_RUNTIME_REQUIRED: two complete 128-cubed native engine constructors, no MD"]
    fn two_native_engines_share_only_immutable_base_threshold() {
        let path = std::env::var("PRISM_IMMUTABLE_OWNER_TEST_TOPOLOGY")
            .expect("set explicit exact-topology fixture path");
        let topology = crate::input::PrismPrepTopology::load(path).unwrap();
        let context = CudaContext::new(0).unwrap();
        let sa = context.new_stream().unwrap();
        let sb = context.new_stream().unwrap();
        let a = crate::fused_engine::NhsAmberFusedEngine::new_on_stream(
            context.clone(),sa.clone(),&topology,128,0.5).unwrap();
        let b = crate::fused_engine::NhsAmberFusedEngine::new_on_stream(
            context,sb.clone(),&topology,128,0.5).unwrap();
        let mut a = CudaOwned::new(a,sa.clone());
        let mut b = CudaOwned::new(b,sb.clone());
        assert_eq!(a.base_threshold_buffer().len(),128usize.pow(3)*8);
        assert_eq!(a.base_threshold_buffer().device_ptr(&sa).0,
                   b.base_threshold_buffer().device_ptr(&sb).0);
        assert_ne!(a.threshold_buffer_mut().device_ptr(&sa).0,
                   b.threshold_buffer_mut().device_ptr(&sb).0);
        assert_eq!(a.d_warp_matrix_n_bytes(),128usize.pow(3)*136);
        assert_eq!(a.d_warp_matrix_dev_ptr(&sa),b.d_warp_matrix_dev_ptr(&sb));
        let expected=sa.clone_dtoh(a.base_threshold_buffer()).unwrap();
        drop(a);
        let retained=sb.clone_dtoh(b.base_threshold_buffer()).unwrap();
        assert!(expected.iter().zip(&retained).all(|(x,y)|x.to_bits()==y.to_bits()));
        drop(b);
        println!("two_native_engines=2 grid=128 immutable_pointer_shared=true mutable_threshold_pointer_shared=false retained_after_first_owner_retirement=true md_steps=0");
    }
}

/// Content-addressed constructor-only host byte publication. Callers retain an
/// independent CudaOwned<Arc<_>> lease per engine, exactly as for the immutable
/// threshold bank. The cache never owns a strong device reference. This is not
/// a writable buffer API; published spatial maps cannot be rebuilt in place.
#[cfg(feature = "gpu")]
pub fn intern_immutable_bytes(
    stream: &Arc<cudarc::driver::CudaStream>, bytes: &[u8],
) -> anyhow::Result<(Arc<cudarc::driver::CudaSlice<u8>>, serde_json::Value)> {
    use anyhow::Context;
    use sha2::Digest;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock, Weak};
    use cudarc::driver::CudaSlice;
    type Key = (usize, usize, [u8;32]);
    static BANKS: OnceLock<Mutex<HashMap<Key,Weak<CudaSlice<u8>>>>> = OnceLock::new();
    anyhow::ensure!(!bytes.is_empty(), "immutable byte publication cannot be empty");
    let digest: [u8;32] = sha2::Sha256::digest(bytes).into();
    let key = (Arc::as_ptr(stream.context()) as usize, bytes.len(), digest);
    let mut banks = BANKS.get_or_init(||Mutex::new(HashMap::new())).lock()
        .map_err(|_|anyhow::anyhow!("immutable byte bank cache poisoned"))?;
    banks.retain(|_,bank|bank.strong_count()>0);
    let (bank,reused) = if let Some(bank)=banks.get(&key).and_then(Weak::upgrade) {
        (bank,true)
    } else {
        let bank=stream.clone_htod(bytes).context("publish exact immutable initialization bytes")?;
        stream.synchronize().context("fence immutable byte publication")?;
        let bank=Arc::new(bank);
        banks.insert(key,Arc::downgrade(&bank));
        (bank,false)
    };
    Ok((bank,serde_json::json!({
        "scope":"same_context_equal_initialized_bytes_immutable_owner",
        "content_sha256":digest.iter().map(|byte|format!("{byte:02x}")).collect::<String>(),
        "allocation_bytes":bytes.len(),
        "reused_existing_allocation":reused,
    })))
}

#[cfg(all(test,feature="gpu"))]
mod immutable_bytes_gpu_tests {
    use super::*;
    use cudarc::driver::{CudaContext,DevicePtr};
    use crate::governor_device::CudaOwned;
    #[test]
    #[ignore = "GPU_RUNTIME_REQUIRED: exact byte banks and stream-retired shared lifetime"]
    fn immutable_byte_banks_preserve_content_and_context_isolation() {
        let ctx=CudaContext::new(0).unwrap();
        let a=ctx.new_stream().unwrap();
        let b=ctx.new_stream().unwrap();
        let other=CudaContext::new(0).unwrap().new_stream().unwrap();
        let bytes:Vec<u8>=(0..8192).map(|i|(i%256) as u8).collect();
        let mut changed=bytes.clone();changed[4097]^=1;
        let (first,_)=intern_immutable_bytes(&a,&bytes).unwrap();
        let (second,receipt)=intern_immutable_bytes(&b,&bytes).unwrap();
        let (different,_)=intern_immutable_bytes(&b,&changed).unwrap();
        let (other_bank,_)=intern_immutable_bytes(&other,&bytes).unwrap();
        assert!(receipt["reused_existing_allocation"].as_bool().unwrap());
        assert!(Arc::ptr_eq(&first,&second));
        assert!(!Arc::ptr_eq(&first,&different));
        assert!(!Arc::ptr_eq(&first,&other_bank));
        assert_eq!(first.device_ptr(&a).0,second.device_ptr(&b).0);
        let weak=Arc::downgrade(&first);
        let owner_a=CudaOwned::new(first,a);
        let owner_b=CudaOwned::new(second,b.clone());
        drop(owner_a);
        assert_eq!(b.clone_dtoh(&**owner_b).unwrap(),bytes);
        drop(owner_b);assert!(weak.upgrade().is_none());
        drop(CudaOwned::new(different,b));
        drop(CudaOwned::new(other_bank,other));
    }
}

#[cfg(all(test,feature="gpu"))]
mod sixteen_constructor_allocation {
    use cudarc::driver::{CudaContext,DevicePtr};
    use crate::governor_device::CudaOwned;
    use std::io::Write;
    #[test]
    #[ignore = "GPU_RUNTIME_REQUIRED: sixteen complete 128-cubed engines retained together, zero MD"]
    fn sixteen_native_engines_full_resolution_allocation() -> anyhow::Result<()> {
        let topology_path=std::env::var("PRISM_IMMUTABLE_OWNER_TEST_TOPOLOGY")?;
        let output=std::env::var("PRISM_SIXTEEN_ALLOCATION_RECEIPT")?;
        let topology=crate::input::PrismPrepTopology::load(&topology_path)?;
        let context=CudaContext::new(0)?;
        let mut writer=std::io::BufWriter::new(std::fs::OpenOptions::new()
            .write(true).create_new(true).open(output)?);
        let mut owners=Vec::new();
        context.bind_to_thread()?;
        let initial=cudarc::driver::result::mem_get_info()?;
        writeln!(writer,"{}",serde_json::json!({"stage":"before_constructors","requested_engines":16,"grid_dim":128,"md_steps":0,"device_global_free_bytes":initial.0,"device_global_total_bytes":initial.1,"topology":topology_path,"n_atoms":topology.n_atoms,"scope":"constructor_capacity_only; protocol_axes_observer_captures_and_dynamic_V2_not_yet_allocated"}))?;
        writer.flush()?;
        let mut base_pointer=None;let mut warp_pointer=None;
        for id in 0..16usize {
            let stream=context.new_stream()?;
            let result=crate::fused_engine::NhsAmberFusedEngine::new_on_stream(
                context.clone(),stream.clone(),&topology,128,0.75);
            let engine=match result {
                Ok(engine)=>engine,
                Err(error)=>{
                    writeln!(writer,"{}",serde_json::json!({"stage":"constructor_failed","stream_id":id,"already_retained_engines":owners.len(),"md_steps":0,"error":format!("{error:#}")}))?;
                    writer.flush()?;
                    return Err(error);
                }
            };
            let owner=CudaOwned::new(engine,stream.clone());
            let base=owner.base_threshold_buffer().device_ptr(&stream).0;
            let warp=owner.d_warp_matrix_dev_ptr(&stream);
            anyhow::ensure!(base!=0 && warp!=0,"unpublished immutable bank");
            if let Some(expected)=base_pointer { anyhow::ensure!(base==expected,"base-bank content/context equality not established"); }
            if let Some(expected)=warp_pointer { anyhow::ensure!(warp==expected,"warp-bank content/context equality not established"); }
            base_pointer=Some(base);warp_pointer=Some(warp);
            owners.push(owner);
            stream.synchronize()?;
            let memory=cudarc::driver::result::mem_get_info()?;
            writeln!(writer,"{}",serde_json::json!({"stage":"constructor_retained","stream_id":id,"retained_engines":owners.len(),"grid_dim":128,"md_steps":0,"device_global_free_bytes":memory.0,"device_global_total_bytes":memory.1,"base_pointer":base,"warp_pointer":warp,"equal_base_bank":true,"equal_warp_bank":true,"scope":"global memory snapshot; no exclusive per-stream allocation attribution"}))?;
            writer.flush()?;
        }
        anyhow::ensure!(owners.len()==16,"all requested owners must coexist");
        context.synchronize()?;
        let memory=cudarc::driver::result::mem_get_info()?;
        writeln!(writer,"{}",serde_json::json!({"stage":"all_sixteen_retained","status":"CONSTRUCTION_VERIFIED","retained_engines":16,"grid_dim":128,"md_steps":0,"device_global_free_bytes":memory.0,"device_global_total_bytes":memory.1,"dynamic_V2_capacity_verified":false,"physical_conditions_configured":false}))?;
        writer.flush()?;
        drop(owners);
        context.synchronize()?;
        let memory=cudarc::driver::result::mem_get_info()?;
        writeln!(writer,"{}",serde_json::json!({"stage":"all_owners_retired","device_global_free_bytes":memory.0,"device_global_total_bytes":memory.1,"md_steps":0}))?;
        writer.flush()?;
        Ok(())
    }
}
