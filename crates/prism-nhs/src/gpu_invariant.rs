//! Rectification Phase 1 — Hard-Trap GPU Invariant Enforcement.
//!
//! Per the PRISM-4D Pipeline Rectification mandate §2 (operator
//! directive 2026-04-29). Replaces soft-assertion error-code returns
//! with hardware-level traps via the PTX `trap` instruction.
//!
//! # Behavior on violation
//!
//! When a device-side `gpu_hard_assert` fires, the warp executing
//! the trap is terminated and the driver delivers a non-recoverable
//! CUDA error. Subsequent stream synchronization on the host
//! returns `cudaErrorIllegalInstruction` (CUDA 13.x driver behavior
//! on Blackwell sm_120) or `cudaErrorLaunchFailure` (older drivers).
//!
//! The CUDA context is poisoned: no further work can be queued
//! without a context reset. This is the §9 ESCALATION path —
//! invariant violation indicates corruption that no amount of
//! host-side recovery can paper over.
//!
//! # Verification posture
//!
//! Two test paths required by mandate §4.1:
//!
//! 1. **Pass path**: synthetic input where the conservation invariant
//!    holds. The audit kernel completes without trapping; stream
//!    synchronize returns success.
//! 2. **Trap path**: synthetic input with a deliberate off-by-one
//!    error so the invariant is violated. Stream synchronize MUST
//!    return a non-success CUDA error. The test runs the trap assertion
//!    in a fresh subprocess so its poisoned CUDA primary context cannot
//!    contaminate the parent test process.

// ============================================================================
// FFI surface
// ============================================================================

#[cfg(feature = "gpu")]
#[allow(dead_code)]
mod ffi {
    pub type CudaError = i32;
    pub const CUDA_SUCCESS: CudaError = 0;

    extern "C" {
        /// Sentinel `0xA55E47`. Confirms the static archive linked
        /// correctly and the FFI ABI is round-tripping.
        pub fn prism_gpu_invariant_link_probe() -> u32;

        /// Launch the M1 Conservation-of-Mass audit kernel on
        /// `stream`. The kernel hard-asserts
        /// `*d_total_attributed + *d_background_count ==
        ///  expected_total_input_spikes` via PTX `trap`. Returns
        /// `cudaSuccess` if the kernel was queued; the caller must
        /// then drive the stream to completion to observe the
        /// trap (or its absence).
        pub fn prism_audit_mass_conservation(
            d_total_attributed: *const u64,
            d_background_count: *const u64,
            expected_total_input_spikes: u64,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;
    }
}

/// Safe wrapper over the FFI link-probe. Returns `0xA55E47`.
#[cfg(feature = "gpu")]
pub fn link_probe() -> u32 {
    unsafe { ffi::prism_gpu_invariant_link_probe() }
}

// ============================================================================
// Safe Rust wrapper
// ============================================================================

/// Launch the M1 Conservation-of-Mass audit kernel.
///
/// On the given stream, reads the two device-side u64 counters,
/// sums them, and hard-asserts equality with `expected_total_input_spikes`
/// via the PTX `trap` instruction. If the assertion fails, the
/// caller observes a non-success error from the next
/// `stream.synchronize()` (or any subsequent stream operation).
///
/// **Returns**:
/// - `Ok(())` if the kernel queue was accepted (NOT proof the audit
///   passed — caller must synchronize).
/// - `Err(CudaError)` if the queue itself failed.
///
/// **Caller contract**: invoke this AFTER the M1 producer has
/// finished writing the conservation scalars. The kernel reads via
/// volatile cast to defeat any compiler load-fusion across prior
/// atomic ops; the stream-order is the source of read-after-write
/// safety.
#[cfg(feature = "gpu")]
pub fn audit_mass_conservation(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    d_total_attributed: u64,
    d_background_count: u64,
    expected_total_input_spikes: u64,
) -> Result<(), i32> {
    let raw_stream = stream.cu_stream() as usize;
    let rc = unsafe {
        ffi::prism_audit_mass_conservation(
            d_total_attributed as *const u64,
            d_background_count as *const u64,
            expected_total_input_spikes,
            raw_stream as *mut std::ffi::c_void,
        )
    };
    if rc != ffi::CUDA_SUCCESS {
        return Err(rc);
    }
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[test]
    fn link_probe_returns_sentinel() {
        // Confirms gpu_invariant.cu linked into the static archive
        // and the FFI ABI is round-tripping. Sentinel pinned at
        // 0xA55E47.
        assert_eq!(super::link_probe(), 0x00A5_5E47);
    }

    #[cfg(feature = "gpu")]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn audit_passes_on_valid_conservation() {
        // ── Pass path verification per mandate §4.1.
        //
        // Synthetic conservation: total_attributed=80, background=20,
        // expected=100. The invariant holds; the audit kernel does
        // NOT trap; stream synchronize returns success.

        use cudarc::driver::CudaContext;
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[audit-pass] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream");

        // Allocate two u64 buffers and htod the values.
        let mut d_total = stream.alloc_zeros::<u64>(1).expect("alloc total");
        let mut d_bg = stream.alloc_zeros::<u64>(1).expect("alloc bg");
        stream
            .memcpy_htod(&[80u64], &mut d_total)
            .expect("htod total");
        stream.memcpy_htod(&[20u64], &mut d_bg).expect("htod bg");

        use cudarc::driver::DevicePtr;
        let (total_dev, _g1) = d_total.device_ptr(&stream);
        let (bg_dev, _g2) = d_bg.device_ptr(&stream);

        super::audit_mass_conservation(&stream, total_dev, bg_dev, 100)
            .expect("audit kernel queue");

        // Stream synchronize — invariant holds, no trap, returns
        // success.
        stream
            .synchronize()
            .expect("audit on valid input must not trap");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn audit_traps_on_violated_conservation() {
        let status = std::process::Command::new(
            std::env::current_exe().expect("resolve current test executable"),
        )
        .args([
            "--exact",
            "gpu_invariant::tests::audit_trap_subprocess_helper",
            "--nocapture",
        ])
        .env("PRISM_GPU_INVARIANT_TRAP_SUBPROCESS", "1")
        .status()
        .expect("run GPU invariant trap subprocess");

        assert!(
            status.success(),
            "GPU invariant trap subprocess failed with status {status}"
        );

        // Prove the child trap did not poison CUDA state in this parent
        // process. On GPU-less builders the child and parent retain the
        // existing skip behavior.
        use cudarc::driver::CudaContext;
        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("[audit-trap-parent] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("parent liveness stream");
        let _sentinel = stream
            .alloc_zeros::<u64>(1)
            .expect("parent CUDA allocation after child trap");
        stream
            .synchronize()
            .expect("parent CUDA context must remain usable after child trap");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn audit_trap_subprocess_helper() {
        if std::env::var_os("PRISM_GPU_INVARIANT_TRAP_SUBPROCESS").is_none() {
            return;
        }

        // ── Trap path verification per mandate §4.1.
        //
        // Synthetic conservation VIOLATED: total_attributed=80,
        // background=20, expected=101 (off by one). The PTX trap
        // fires; stream synchronize returns a CUDA error
        // (cudaErrorIllegalInstruction or cudaErrorLaunchFailure
        // depending on driver).
        //
        // The CUDA context is POISONED after the trap. This helper is selected
        // by exact test name in a fresh process, so process teardown destroys
        // the poisoned primary-context client before the parent suite resumes.

        use cudarc::driver::CudaContext;
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[audit-trap] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream");

        let mut d_total = stream.alloc_zeros::<u64>(1).expect("alloc total");
        let mut d_bg = stream.alloc_zeros::<u64>(1).expect("alloc bg");
        stream
            .memcpy_htod(&[80u64], &mut d_total)
            .expect("htod total");
        stream.memcpy_htod(&[20u64], &mut d_bg).expect("htod bg");

        use cudarc::driver::DevicePtr;
        let (total_dev, _g1) = d_total.device_ptr(&stream);
        let (bg_dev, _g2) = d_bg.device_ptr(&stream);

        super::audit_mass_conservation(&stream, total_dev, bg_dev, /* off-by-one */ 101)
            .expect("audit kernel queue (queue itself must accept; trap fires at exec)");

        // Stream synchronize MUST return an error. This is the
        // hard-trap signature on Blackwell sm_120: the kernel
        // terminated via PTX trap, and CUDA's driver propagates
        // the violation as cudaErrorIllegalInstruction (or
        // cudaErrorLaunchFailure on older drivers). Either is
        // acceptable here — we only assert that the synchronize
        // does NOT return success.
        let sync_result = stream.synchronize();
        match sync_result {
            Ok(()) => panic!(
                "PTX trap path FAILED: stream.synchronize() returned Ok on a violated invariant. \
                 Expected cudaErrorIllegalInstruction or cudaErrorLaunchFailure. \
                 The hard-trap idiom is not firing — invariant enforcement is BROKEN."
            ),
            Err(e) => {
                // Print for forensic visibility but do NOT fail the
                // test. Any error from synchronize is the expected
                // post-trap state.
                eprintln!(
                    "[audit-trap] Hard-trap fired as expected. stream.synchronize() error: {:?}",
                    e
                );
            }
        }
    }
}
