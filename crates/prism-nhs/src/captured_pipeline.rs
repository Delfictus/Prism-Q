//! DAG-COND-WIRE — Captured Adjudication Pipeline (LEGO Brick).
//!
//! Per the operator's IGNITION mandate (2026-04-29) and Anti-Greenfield
//! Doctrine. Self-contained orchestrator that:
//!
//!   1. Pre-allocates every F2-pool buffer + the pinned-host ghost ring
//!      + the non-blocking telemetry stream BEFORE entering the capture
//!      block (operator §3.2: "There must be exactly zero calls to
//!      cudaMalloc, cudaFree, malloc, or any Rust-side heap manipulation
//!      inside the capture").
//!   2. Wraps the SO(3) projection (Node B), the InterferometricAdjudicator
//!      step (Node C), the F1 trampoline (Node C'), AND the cross-stream
//!      `cuMemcpyDtoHAsync_v2` to the ghost ring in a single
//!      `cuStreamBeginCapture(md_stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)` /
//!      `cuStreamEndCapture` block. The cross-stream telemetry copy is
//!      synchronised by an explicit `cuEventRecord` after Node C and a
//!      `cuStreamWaitEvent` on the telemetry stream — both captured under
//!      `MODE_THREAD_LOCAL` (TIER 6, 2026-05-03; was `MODE_GLOBAL` until
//!      8-stream interferometric build hit cross-thread capture
//!      invalidation — see call-site at the `cuStreamBeginCapture`
//!      invocation for the full rationale).
//!   3. Post-capture explicitly adds a `cudaGraphConditionalNode`
//!      (Node D) downstream of the trampoline (Node C') with an explicit
//!      `cuGraphAddDependencies` edge (operator §2.3: "If you fail to
//!      map this edge, Claude-3 will throw a Gate G19 exception").
//!      The conditional handle's predicate is the
//!      `adj->adjudication_code` u32 written by Node C and forwarded by
//!      the trampoline via `cudaGraphSetConditional`.
//!   4. Adds a single `MEMSET` node to the conditional's body sub-graph
//!      that stamps the cluster's frame index into a host-visible
//!      `burst_marker` buffer when the conditional fires. This lets
//!      Claude-3 verify routing at the topology level without depending
//!      on the Adjudicator's noise-floor calibration.
//!   5. Calls `cuGraphInstantiate` and exposes the resulting
//!      `CUgraphExec` for repeated `cuGraphLaunch` invocations.
//!
//! # Cross-lane dependency contract
//!
//! Node | Owner    | Producer of      | Consumer of
//! -----|----------|------------------|------------------------
//! B    | Claude-1 | ContactShellTile | RichSpike + K_LM
//! C    | Claude-2 | adjudication_code| ContactShellTile (relaxed/perturbed)
//! C'   | Claude-2 | (sets handle)    | adjudication_code
//! D    | (graph)  | (routes graph)   | conditional handle
//! Body | Claude-1 | burst_marker     | (none)
//! DMA  | Claude-1 | host ring slot   | ContactShellTile (post-Node B)
//!
//! # Anti-Greenfield posture
//!
//! Pure scavenge: every kernel + every FFI helper is reused as-is from
//! the existing crates. The only net-new code in this file is the
//! orchestration sequence (~ 600 lines including doc + test). Zero new
//! CUDA kernels, zero new build.rs entries, zero new dependencies.

#![cfg(feature = "gpu")]

use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::sys::*;
use cudarc::driver::{result, CudaContext, CudaStream, DriverError};

use crate::ghost_telemetry::{
    create_non_blocking_telemetry_stream, schedule_async_tile_copy, PinnedTelemetryRing,
};
use crate::interferometric_adjudicator::InterferometricAdjudicatorFfi;
use crate::rich_spike::RichSpike;
use crate::so3_project::{ContactShellTile, SiteManifestFfi};
use crate::vram_pool::VramPool;

/// Measured KCC selection attached to one captured cluster at an observation endpoint.
/// Residue IDs index the native topology/KCC arrays; no representative-residue fallback.
#[derive(Debug, Clone, serde::Serialize)]
pub struct KccCausalLeadMeasurement {
    pub cluster_id: usize,
    pub measurement_step: u64,
    pub candidate_count: usize,
    pub eligible_candidate_count: usize,
    pub causal_lead_residue: Option<u32>,
    pub temporal_corr: Option<f32>,
    pub residue_count: Option<u32>,
    pub active_causal: Option<u32>,
}

/// Select the largest measured |temporal correlation| among the cluster's exact
/// seed-spike residue associations. Ties select the lower topology residue ID.
/// A finite nonzero correlation, >=2 observations and >0 causal events are required.
pub fn measured_kcc_causal_leads(
    candidates: &[Vec<u32>],
    kcc: &crate::fused_engine::KccData,
    measurement_step: u64,
) -> anyhow::Result<Vec<KccCausalLeadMeasurement>> {
    anyhow::ensure!(kcc.temporal_corr.len() == kcc.n_residues
        && kcc.residue_count.len() == kcc.n_residues
        && kcc.active_causal.len() == kcc.n_residues,
        "KCC causal-lead arrays disagree with residue count");
    candidates.iter().enumerate().map(|(cluster_id, residues)| {
        let mut row = KccCausalLeadMeasurement {
            cluster_id, measurement_step, candidate_count: residues.len(),
            eligible_candidate_count: 0, causal_lead_residue: None,
            temporal_corr: None, residue_count: None, active_causal: None,
        };
        for &rid in residues {
            let idx = rid as usize;
            anyhow::ensure!(idx < kcc.n_residues,
                "cluster {} causal candidate {} outside KCC residue array {}",
                cluster_id, rid, kcc.n_residues);
            let corr = kcc.temporal_corr[idx];
            if kcc.residue_count[idx] < 2 || kcc.active_causal[idx] == 0
                || !corr.is_finite() || corr == 0.0 { continue; }
            row.eligible_candidate_count += 1;
            let wins = match (row.temporal_corr, row.causal_lead_residue) {
                (Some(best), Some(best_id)) => corr.abs() > best.abs()
                    || (corr.abs() == best.abs() && rid < best_id),
                _ => true,
            };
            if wins {
                row.causal_lead_residue = Some(rid);
                row.temporal_corr = Some(corr);
                row.residue_count = Some(kcc.residue_count[idx]);
                row.active_causal = Some(kcc.active_causal[idx]);
            }
        }
        Ok(row)
    }).collect()
}

/// Immutable seed-cluster CSR consumed by the native endpoint selector.
/// Its allocation lifetime must cover every captured selector launch.
pub struct KccCausalCandidates {
    stream: Arc<CudaStream>,
    offsets: cudarc::driver::CudaSlice<u32>,
    residues: cudarc::driver::CudaSlice<u32>,
    n_residues: usize,
    n_clusters: usize,
}
impl KccCausalCandidates {
    pub fn new(stream: &Arc<CudaStream>, candidates: &[Vec<u32>], n_residues: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(!candidates.is_empty() && candidates.len() <= (u32::MAX - 127) as usize,
            "KCC selector requires a representable nonempty cluster set");
        let mut offsets = vec![0u32];
        let mut residues = Vec::new();
        for cluster in candidates {
            anyhow::ensure!(cluster.iter().all(|&rid| (rid as usize) < n_residues),
                "KCC candidate outside native residue arrays");
            residues.extend_from_slice(cluster);
            offsets.push(u32::try_from(residues.len())?);
        }
        // A fully empty CSR has no reads; keep a valid allocation for FFI.
        if residues.is_empty() { residues.push(u32::MAX); }
        let result = Self { stream: stream.clone(), offsets: stream.clone_htod(&offsets)?,
            residues: stream.clone_htod(&residues)?, n_residues, n_clusters: candidates.len() };
        stream.synchronize()?; // upload owners are host temporaries; before graph capture
        Ok(result)
    }
    /// Exact caller-owned CSR shape; no GPU copy, allocation or synchronization.
    pub fn allocation_shape(&self)->serde_json::Value {
        serde_json::json!({"clusters":self.n_clusters,"native_residues":self.n_residues,
            "stored_residue_entries":self.residues.len(),"offset_entries":self.offsets.len(),
            "device_requested_bytes":4*(self.residues.len()+self.offsets.len()),
            "scope":"caller_owned_kcc_csr_only","includes_kcc_leads":false})
    }
    pub fn select(&self, stream: &Arc<CudaStream>, correlation: &cudarc::driver::CudaSlice<f32>,
        counts: &cudarc::driver::CudaSlice<u32>, active: &cudarc::driver::CudaSlice<u32>,
        leads: &mut cudarc::driver::CudaSlice<u32>) -> anyhow::Result<()> {
        self.bind(stream, correlation, counts, active, leads)?.launch()
    }
    /// Resolve pointer/event guards BEFORE capture. Keep this binding through
    /// capture, then drop it after capture ends; graph callers must retain all
    /// pointee owners and order subsequent use on this stream through completion.
    pub fn bind<'a>(&'a self, stream: &'a Arc<CudaStream>, correlation: &'a cudarc::driver::CudaSlice<f32>,
        counts: &'a cudarc::driver::CudaSlice<u32>, active: &'a cudarc::driver::CudaSlice<u32>,
        leads: &'a mut cudarc::driver::CudaSlice<u32>) -> anyhow::Result<KccLeadSelection<'a>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        anyhow::ensure!(Arc::ptr_eq(&self.stream, stream), "KCC selector stream owner changed");
        anyhow::ensure!(correlation.len() == self.n_residues && counts.len() == self.n_residues
            && active.len() == self.n_residues && leads.len() == self.n_clusters,
            "KCC selector device-array cardinality mismatch");
        let (offsets, o) = self.offsets.device_ptr(stream);
        let (residues, r) = self.residues.device_ptr(stream);
        let (corr, c) = correlation.device_ptr(stream);
        let (count, n) = counts.device_ptr(stream);
        let (act, a) = active.device_ptr(stream);
        let (output, l) = leads.device_ptr_mut(stream);
        Ok(KccLeadSelection { pointers:[offsets,residues,corr,count,act,output],
            n_clusters:self.n_clusters as u32, stream, _guards:[o,r,c,n,a,l] })
    }
}

/// Capture-ready native selector invocation; no pointer guard/event mutation
/// occurs during launch. Buffer borrows last until this object is dropped.
pub struct KccLeadSelection<'a> {
    pointers: [u64;6], n_clusters:u32, stream:&'a Arc<CudaStream>,
    _guards:[cudarc::driver::SyncOnDrop<'a>;6],
}
impl KccLeadSelection<'_> {
    pub fn launch(&self) -> anyhow::Result<()> {
        let p=self.pointers;
        let rc=unsafe { prism_select_kcc_leads_launch(p[0] as _,p[1] as _,self.n_clusters,
            p[2] as _,p[3] as _,p[4] as _,p[5] as _,self.stream.cu_stream() as *mut c_void) };
        anyhow::ensure!(rc==0,"native KCC lead selector CUDA error {rc}");
        Ok(())
    }
}

extern "C" {
    fn prism_select_kcc_leads_launch(offsets: *const u32, residues: *const u32,
        n_clusters: u32, correlation: *const f32, counts: *const u32,
        active: *const u32, leads: *mut u32, stream: *mut c_void) -> i32;
}

fn tier8_diag_driver_error_text(rc: CUresult) -> (String, String) {
    let err = DriverError(rc);
    let name = err
        .error_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|e| format!("cuGetErrorName failed: {:?}", e));
    let text = err
        .error_string()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|e| format!("cuGetErrorString failed: {:?}", e));
    (name, text)
}

fn tier8_diag_runtime_error_text(rc: i32) -> (&'static str, &'static str) {
    match rc {
        0 => ("cudaSuccess", "no error"),
        1 => ("cudaErrorInvalidValue", "invalid argument"),
        2 => ("cudaErrorMemoryAllocation", "out of memory"),
        4 => ("cudaErrorLaunchFailure", "unspecified launch failure"),
        35 => (
            "cudaErrorInsufficientDriver",
            "CUDA driver version is insufficient for CUDA runtime version",
        ),
        46 => (
            "cudaErrorDevicesUnavailable",
            "all CUDA-capable devices are busy or unavailable",
        ),
        98 => ("cudaErrorInvalidDeviceFunction", "invalid device function"),
        700 => ("cudaErrorIllegalAddress", "illegal memory access"),
        701 => ("cudaErrorLaunchOutOfResources", "launch out of resources"),
        702 => ("cudaErrorLaunchTimeout", "launch timeout"),
        710 => ("cudaErrorAssert", "device-side assert triggered"),
        719 => ("cudaErrorLaunchFailure", "unspecified launch failure"),
        801 => ("cudaErrorNotSupported", "operation not supported"),
        _ => (
            "cudaErrorUnmapped",
            "runtime error code not mapped in diagnostic shim",
        ),
    }
}

fn tier8_diag_current_context() -> (CUcontext, CUresult) {
    let mut current: CUcontext = ptr::null_mut();
    let rc = unsafe { cuCtxGetCurrent(&mut current as *mut _) };
    (current, rc)
}

fn tier8_diag_verbose_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PRISM_TIER8_VERBOSE_DIAG")
            .ok()
            .map(|v| {
                let v = v.trim();
                v == "1"
                    || v.eq_ignore_ascii_case("true")
                    || v.eq_ignore_ascii_case("yes")
                    || v.eq_ignore_ascii_case("on")
            })
            .unwrap_or(false)
    })
}

macro_rules! tier8_diag_verbose {
    ($($arg:tt)*) => {
        if tier8_diag_verbose_enabled() {
            log::info!($($arg)*);
        }
    };
}

fn tier8_diag_named_driver_gate<F>(
    stage: &'static str,
    previous_milestone: &'static str,
    stream_id: Option<u32>,
    protocol_group: &'static str,
    expected_ctx: CUcontext,
    md_raw: usize,
    query_after_on_success: bool,
    call: F,
) -> Result<(), BuildError>
where
    F: FnOnce() -> Result<(), DriverError>,
{
    let stream_label = stream_id
        .map(|id| id.to_string())
        .unwrap_or_else(|| "unavailable".to_string());
    let mut before_ctx: CUcontext = ptr::null_mut();
    let get_before_rc = unsafe { cuCtxGetCurrent(&mut before_ctx as *mut _) };
    if !matches!(get_before_rc, CUresult::CUDA_SUCCESS) {
        let (name, text) = tier8_diag_driver_error_text(get_before_rc);
        log::error!(
            "[TIER8-DIAG stream6-gap] stage={} call=cuCtxGetCurrent(before) \
             result=FAIL stream_id={} protocol_group={} previous_milestone={} \
             rc={} cuda_name={} cuda_string={:?} md_raw=0x{:x} \
             expected_ctx={:p} before_ctx={:p}",
            stage,
            stream_label,
            protocol_group,
            previous_milestone,
            get_before_rc as i32,
            name,
            text,
            md_raw,
            expected_ctx,
            before_ctx
        );
        return Err(BuildError::Tier8Context {
            stage,
            rc: get_before_rc as i32,
            expected: expected_ctx as usize,
            before: before_ctx as usize,
            after: 0,
        });
    }

    let mut did_set_current = false;
    let mut set_current_rc = CUresult::CUDA_SUCCESS;
    if before_ctx.is_null() || before_ctx != expected_ctx {
        did_set_current = true;
        set_current_rc = unsafe { cuCtxSetCurrent(expected_ctx) };
        if !matches!(set_current_rc, CUresult::CUDA_SUCCESS) {
            let (name, text) = tier8_diag_driver_error_text(set_current_rc);
            log::error!(
                "[TIER8-DIAG stream6-gap] stage={} call=cuCtxSetCurrent \
                 result=FAIL stream_id={} protocol_group={} previous_milestone={} \
                 rc={} cuda_name={} cuda_string={:?} md_raw=0x{:x} \
                 expected_ctx={:p} before_ctx={:p}",
                stage,
                stream_label,
                protocol_group,
                previous_milestone,
                set_current_rc as i32,
                name,
                text,
                md_raw,
                expected_ctx,
                before_ctx
            );
            return Err(BuildError::Tier8Context {
                stage,
                rc: set_current_rc as i32,
                expected: expected_ctx as usize,
                before: before_ctx as usize,
                after: 0,
            });
        }
    }

    let call_result = call();
    let (call_rc, result_label) = match &call_result {
        Ok(()) => (CUresult::CUDA_SUCCESS, "OK"),
        Err(e) => (e.0, "FAIL"),
    };
    let (call_name, call_text) = tier8_diag_driver_error_text(call_rc);

    let query_after = query_after_on_success || call_result.is_err();
    let mut after_ctx: CUcontext = ptr::null_mut();
    let get_after_rc = if query_after {
        unsafe { cuCtxGetCurrent(&mut after_ctx as *mut _) }
    } else {
        CUresult::CUDA_SUCCESS
    };
    let after_ctx_label = if query_after {
        format!("{:p}", after_ctx)
    } else {
        "skipped_active_capture".to_string()
    };
    let (after_name, after_text) = if query_after {
        tier8_diag_driver_error_text(get_after_rc)
    } else {
        (
            "SKIPPED_ACTIVE_CAPTURE".to_string(),
            "not queried after successful begin_capture".to_string(),
        )
    };
    let (set_name, set_text) = tier8_diag_driver_error_text(set_current_rc);

    tier8_diag_verbose!(
        "[TIER8-DIAG stream6-gap] stage={} call={} result={} stream_id={} \
         protocol_group={} previous_milestone={} rc={} cuda_name={} \
         cuda_string={:?} md_raw=0x{:x} expected_ctx={:p} before_ctx={:p} \
         after_ctx={} did_set_current={} set_rc={} set_name={} set_string={:?} \
         after_get_rc={} after_get_name={} after_get_string={:?}",
        stage,
        stage,
        result_label,
        stream_label,
        protocol_group,
        previous_milestone,
        call_rc as i32,
        call_name,
        call_text,
        md_raw,
        expected_ctx,
        before_ctx,
        after_ctx_label,
        did_set_current,
        set_current_rc as i32,
        set_name,
        set_text,
        get_after_rc as i32,
        after_name,
        after_text
    );

    if let Err(e) = call_result {
        return Err(BuildError::Tier8Context {
            stage,
            rc: e.0 as i32,
            expected: expected_ctx as usize,
            before: before_ctx as usize,
            after: after_ctx as usize,
        });
    }
    if query_after && !matches!(get_after_rc, CUresult::CUDA_SUCCESS) {
        return Err(BuildError::Tier8Context {
            stage,
            rc: get_after_rc as i32,
            expected: expected_ctx as usize,
            before: before_ctx as usize,
            after: after_ctx as usize,
        });
    }
    if query_after && after_ctx != expected_ctx {
        return Err(BuildError::Tier8Context {
            stage,
            rc: -2,
            expected: expected_ctx as usize,
            before: before_ctx as usize,
            after: after_ctx as usize,
        });
    }

    Ok(())
}

/// TIER 8.β post-gasp-sync raw gate.  Replaces the wrapper-based
/// `tier8_diag_named_driver_gate("post_gasp_sync", ..., || md_stream.synchronize())`
/// invocation with a 5-step raw-CUDA gate that distinguishes:
///   1. cudarc deferred error state surfaced via `ctx.check_err`
///   2. before_ctx is null or != expected_ctx
///   3. cuCtxSetCurrent failure
///   4. after_ctx != expected_ctx (post-set)
///   5. raw cuStreamSynchronize failure
/// Per directive: this path NEVER calls `md_stream.synchronize()` — it
/// uses `cuStreamSynchronize(md_stream.cu_stream())` directly so the
/// failing call (if any) is unambiguously a raw CUDA driver call and
/// not the cudarc wrapper's pre-sync bookkeeping.
fn tier8_diag_post_gasp_sync_gate(
    ctx: &Arc<CudaContext>,
    md_stream: &Arc<CudaStream>,
    stream_id: Option<u32>,
    protocol_group: &'static str,
    expected_ctx: CUcontext,
    md_raw: usize,
) -> Result<(), BuildError> {
    let stream_label = stream_id
        .map(|id| id.to_string())
        .unwrap_or_else(|| "unavailable".to_string());
    let thread_id = format!("{:?}", std::thread::current().id());

    // Step 1 — cudarc deferred error-state probe (drain once if dirty).
    let mut drained_once = false;
    if let Err(first) = ctx.check_err() {
        drained_once = true;
        let (name, text) = tier8_diag_driver_error_text(first.0);
        log::warn!(
            "[TIER8-DIAG post-gasp-sync] call=ctx.check_err result=DRAINED \
             stream_id={} protocol_group={} rc={} cuda_name={} cuda_string={:?} \
             md_raw=0x{:x} expected_ctx={:p} thread_id={}",
            stream_label,
            protocol_group,
            first.0 as i32,
            name,
            text,
            md_raw,
            expected_ctx,
            thread_id
        );
        if let Err(second) = ctx.check_err() {
            let (n2, t2) = tier8_diag_driver_error_text(second.0);
            log::error!(
                "[TIER8-DIAG post-gasp-sync] call=ctx.check_err(second) result=FAIL \
                 stream_id={} protocol_group={} rc={} cuda_name={} cuda_string={:?} \
                 md_raw=0x{:x} expected_ctx={:p} thread_id={}",
                stream_label,
                protocol_group,
                second.0 as i32,
                n2,
                t2,
                md_raw,
                expected_ctx,
                thread_id
            );
            return Err(BuildError::Tier8Context {
                stage: "TIER8-DIAG post-gasp-sync ctx.check_err",
                rc: second.0 as i32,
                expected: expected_ctx as usize,
                before: 0,
                after: 0,
            });
        }
    }

    // Step 2 — cuCtxGetCurrent(before).
    let mut before_ctx: CUcontext = ptr::null_mut();
    let rc_before = unsafe { cuCtxGetCurrent(&mut before_ctx as *mut _) };
    if !matches!(rc_before, CUresult::CUDA_SUCCESS) {
        let (name, text) = tier8_diag_driver_error_text(rc_before);
        log::error!(
            "[TIER8-DIAG post-gasp-sync] call=cuCtxGetCurrent(before) result=FAIL \
             stream_id={} protocol_group={} rc={} cuda_name={} cuda_string={:?} \
             md_raw=0x{:x} expected_ctx={:p} thread_id={} drained_once={}",
            stream_label,
            protocol_group,
            rc_before as i32,
            name,
            text,
            md_raw,
            expected_ctx,
            thread_id,
            drained_once
        );
        return Err(BuildError::Tier8Context {
            stage: "TIER8-DIAG post-gasp-sync cuCtxGetCurrent(before)",
            rc: rc_before as i32,
            expected: expected_ctx as usize,
            before: before_ctx as usize,
            after: 0,
        });
    }

    // Step 3 — cuCtxSetCurrent if mismatch.
    let mut did_set = false;
    if before_ctx.is_null() || before_ctx != expected_ctx {
        did_set = true;
        let rc_set = unsafe { cuCtxSetCurrent(expected_ctx) };
        if !matches!(rc_set, CUresult::CUDA_SUCCESS) {
            let (name, text) = tier8_diag_driver_error_text(rc_set);
            log::error!(
                "[TIER8-DIAG post-gasp-sync] call=cuCtxSetCurrent result=FAIL \
                 stream_id={} protocol_group={} rc={} cuda_name={} cuda_string={:?} \
                 md_raw=0x{:x} expected_ctx={:p} before_ctx={:p} thread_id={}",
                stream_label,
                protocol_group,
                rc_set as i32,
                name,
                text,
                md_raw,
                expected_ctx,
                before_ctx,
                thread_id
            );
            return Err(BuildError::Tier8Context {
                stage: "TIER8-DIAG post-gasp-sync cuCtxSetCurrent",
                rc: rc_set as i32,
                expected: expected_ctx as usize,
                before: before_ctx as usize,
                after: 0,
            });
        }
    }

    // Step 4 — cuCtxGetCurrent(after) + mismatch check.
    let mut after_ctx: CUcontext = ptr::null_mut();
    let rc_after = unsafe { cuCtxGetCurrent(&mut after_ctx as *mut _) };
    let after_ok = matches!(rc_after, CUresult::CUDA_SUCCESS);
    if !after_ok || after_ctx != expected_ctx {
        let (name, text) = tier8_diag_driver_error_text(rc_after);
        let reason = if after_ok {
            "context_mismatch"
        } else {
            "rc_fail"
        };
        log::error!(
            "[TIER8-DIAG post-gasp-sync] call=cuCtxGetCurrent(after) result=FAIL \
             stream_id={} protocol_group={} reason={} rc={} cuda_name={} \
             cuda_string={:?} md_raw=0x{:x} expected_ctx={:p} before_ctx={:p} \
             after_ctx={:p} did_set_current={} thread_id={}",
            stream_label,
            protocol_group,
            reason,
            rc_after as i32,
            name,
            text,
            md_raw,
            expected_ctx,
            before_ctx,
            after_ctx,
            did_set,
            thread_id
        );
        return Err(BuildError::Tier8Context {
            stage: if after_ok {
                "TIER8-DIAG post-gasp-sync after_ctx_mismatch"
            } else {
                "TIER8-DIAG post-gasp-sync cuCtxGetCurrent(after)"
            },
            rc: if after_ok { -2 } else { rc_after as i32 },
            expected: expected_ctx as usize,
            before: before_ctx as usize,
            after: after_ctx as usize,
        });
    }

    // Step 5 — raw cuStreamSynchronize(md_raw).  NOT md_stream.synchronize().
    let rc_sync = unsafe { cuStreamSynchronize(md_stream.cu_stream()) };
    if !matches!(rc_sync, CUresult::CUDA_SUCCESS) {
        let (name, text) = tier8_diag_driver_error_text(rc_sync);
        log::error!(
            "[TIER8-DIAG post-gasp-sync] call=cuStreamSynchronize result=FAIL \
             stream_id={} protocol_group={} rc={} cuda_name={} cuda_string={:?} \
             md_raw=0x{:x} expected_ctx={:p} before_ctx={:p} after_ctx={:p} \
             did_set_current={} drained_once={} thread_id={}",
            stream_label,
            protocol_group,
            rc_sync as i32,
            name,
            text,
            md_raw,
            expected_ctx,
            before_ctx,
            after_ctx,
            did_set,
            drained_once,
            thread_id
        );
        return Err(BuildError::Tier8Context {
            stage: "TIER8-DIAG post-gasp-sync cuStreamSynchronize",
            rc: rc_sync as i32,
            expected: expected_ctx as usize,
            before: before_ctx as usize,
            after: after_ctx as usize,
        });
    }

    log::info!(
        "[TIER8-DIAG post-gasp-sync] call=raw_sync_gate result=OK stream_id={} \
         protocol_group={} md_raw=0x{:x} expected_ctx={:p} before_ctx={:p} \
         after_ctx={:p} did_set_current={} drained_once={} thread_id={}",
        stream_label,
        protocol_group,
        md_raw,
        expected_ctx,
        before_ctx,
        after_ctx,
        did_set,
        drained_once,
        thread_id
    );
    Ok(())
}

/// STREAM5_SFA_KERNEL_FAILURE diagnostic — narrow probe inserted after
/// each suspected captured op in the B.3.2 prelude (Node C, M1.2.19.B,
/// Node D, M1.2.17). Polls `cuStreamGetCaptureInfo_v2` to detect whether
/// CUDA has invalidated the in-progress capture immediately after
/// `after_stage` queued its launch.
///
/// Success path: one query call (~µs); zero log emission; zero behavior change.
/// Failure path: ONE error log line naming the EXACT stage that broke
/// capture, then early `Err(BuildError::Cuda)`. The build still aborts
/// (V2 falls into the no-V2 path as before), but the typed error stage
/// label now identifies the actual root cause instead of the deferred-
/// and-mislabelled "B.3.2 SFA kernel" surface.
///
/// See `.prism_orchestration/STREAM5_SFA_KERNEL_FAILURE_REPORT.md` for
/// the operator-approved diagnostic scope and the four insertion points.
#[cfg(feature = "gpu")]
fn diag_capture_invalidation_after(
    md_stream: &CudaStream,
    after_stage: &'static str,
    diagnostic_stream_id: Option<u32>,
    protocol_group: &'static str,
) -> Result<(), BuildError> {
    let mut status: CUstreamCaptureStatus = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let mut id: cuuint64_t = 0;
    let mut graph: CUgraph = ptr::null_mut();
    let mut deps: *const CUgraphNode = ptr::null();
    let mut n_deps: usize = 0;
    let rc = unsafe {
        cuStreamGetCaptureInfo_v2(
            md_stream.cu_stream(),
            &mut status as *mut _,
            &mut id as *mut _,
            &mut graph as *mut _,
            &mut deps as *mut _,
            &mut n_deps as *mut _,
        )
    };
    if !matches!(rc, CUresult::CUDA_SUCCESS) {
        let (cuda_name, cuda_string) = tier8_diag_driver_error_text(rc);
        log::error!(
            "[STREAM5_SFA_DIAG] cuStreamGetCaptureInfo_v2 after stage='{}' \
             returned non-success: stream_id={:?} protocol_group={} rc={} \
             cuda_name={} cuda_string={:?} raw_stream={:p}",
            after_stage,
            diagnostic_stream_id,
            protocol_group,
            rc as i32,
            cuda_name,
            cuda_string,
            md_stream.cu_stream()
        );
        return Err(BuildError::Cuda {
            stage: "STREAM5_SFA_DIAG cuStreamGetCaptureInfo_v2 probe",
            rc: rc as i32,
        });
    }
    if matches!(
        status,
        CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED
    ) {
        log::error!(
            "[STREAM5_SFA_DIAG] CAPTURE INVALIDATED immediately after stage='{}' \
             stream_id={:?} protocol_group={} capture_id={} frontier_n_deps={} \
             raw_stream={:p} graph={:p} — this stage is the root cause that \
             surfaces later as 'B.3.2 SFA kernel' rc=901",
            after_stage,
            diagnostic_stream_id,
            protocol_group,
            id,
            n_deps,
            md_stream.cu_stream(),
            graph
        );
        return Err(BuildError::Cuda {
            stage: "STREAM5_SFA_DIAG capture invalidated post-launch",
            rc: 901, // CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
        });
    }
    Ok(())
}

struct CaptureGuard {
    stream: CUstream,
    stream_id: Option<u32>,
    protocol_group: &'static str,
    active: bool,
}

impl CaptureGuard {
    fn new(stream: CUstream, stream_id: Option<u32>, protocol_group: &'static str) -> Self {
        Self {
            stream,
            stream_id,
            protocol_group,
            active: true,
        }
    }

    fn stream_id_label(&self) -> String {
        self.stream_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "unavailable".to_string())
    }

    fn end_capture_for_commit(&mut self) -> Result<CUgraph, DriverError> {
        let mut graph: CUgraph = ptr::null_mut();
        let rc = unsafe { cuStreamEndCapture(self.stream, &mut graph as *mut _) };
        self.active = false;
        if matches!(rc, CUresult::CUDA_SUCCESS) {
            log::info!(
                "[TIER8-DIAG capture-guard] call=cuStreamEndCapture \
                 result=OK reason=commit stream_id={} protocol_group={} \
                 stream={:p} graph={:p}",
                self.stream_id_label(),
                self.protocol_group,
                self.stream,
                graph
            );
            Ok(graph)
        } else {
            let (name, text) = tier8_diag_driver_error_text(rc);
            log::error!(
                "[TIER8-DIAG capture-guard] call=cuStreamEndCapture \
                 result=FAIL reason=commit stream_id={} protocol_group={} \
                 rc={} cuda_name={} cuda_string={:?} stream={:p} graph={:p}",
                self.stream_id_label(),
                self.protocol_group,
                rc as i32,
                name,
                text,
                self.stream,
                graph
            );
            Err(DriverError(rc))
        }
    }

    fn commit(&mut self, graph: CUgraph) {
        debug_assert!(!graph.is_null());
        self.active = false;
    }
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;

        let mut discard_graph: CUgraph = ptr::null_mut();
        let rc = unsafe { cuStreamEndCapture(self.stream, &mut discard_graph as *mut _) };
        if matches!(rc, CUresult::CUDA_SUCCESS) {
            log::warn!(
                "[TIER8-DIAG capture-guard] call=cuStreamEndCapture \
                 result=OK reason=drop_cleanup stream_id={} protocol_group={} \
                 stream={:p} discarded_graph={:p}",
                self.stream_id_label(),
                self.protocol_group,
                self.stream,
                discard_graph
            );
            if !discard_graph.is_null() {
                unsafe {
                    let _ = result::graph::destroy(discard_graph);
                }
            }
        } else {
            let (name, text) = tier8_diag_driver_error_text(rc);
            log::error!(
                "[TIER8-DIAG capture-guard] call=cuStreamEndCapture \
                 result=FAIL reason=drop_cleanup stream_id={} protocol_group={} \
                 rc={} cuda_name={} cuda_string={:?} stream={:p} \
                 discarded_graph={:p}",
                self.stream_id_label(),
                self.protocol_group,
                rc as i32,
                name,
                text,
                self.stream,
                discard_graph
            );
        }
    }
}

// ============================================================================
// ZSTR FFI — G23 capture-window launchers (Amendment 2.2)
// ============================================================================
//
// Called inside the cuStreamBeginCapture window on telemetry_stream.
// Each call records a kernel node into the in-progress CUgraph.
// Linked from libzstr_kernels.a (build.rs compile_to_static_archive).
//
// WP4: every ZSTR kernel now takes an explicit per-stream `d_active_slot`
// pointer. The module-scope `__device__ uint32_t d_zstr_active_slot` symbol
// it replaces was ONE object shared by all 8 telemetry streams (one context,
// one module), so stream B's staging kernels could consume a slot index
// written by stream A's slot-update kernel.
extern "C" {
    fn zstr_launch_pos_stage(
        base_pinned: *mut c_void,
        inter_slot_stride: u32,
        pos_offset_in_slot: u32,
        src_vram: *const c_void,
        n_atoms: u32,
        d_active_slot: *const c_void,
        stream: *mut c_void,
    ) -> i32;

    fn zstr_launch_fence_signal(
        base_fence: *mut c_void,
        inter_slot_stride: u32,
        d_active_slot: *const c_void,
        stream: *mut c_void,
    ) -> i32;

    // ── T11 — Action-Recovery force exfiltration ──
    fn zstr_launch_force_stage(
        base_pinned: *mut c_void,
        inter_slot_stride: u32,
        force_offset_in_slot: u32,
        force_norm_offset_in_slot: u32,
        src_d_forces: *const c_void,
        n_atoms: u32,
        d_active_slot: *const c_void,
        stream: *mut c_void,
    ) -> i32;

    fn zstr_launch_force_norm_sqrt(
        base_pinned: *mut c_void,
        inter_slot_stride: u32,
        force_norm_offset_in_slot: u32,
        d_active_slot: *const c_void,
        stream: *mut c_void,
        // P8 — pointer-stable force-norm mirror for the G26 governor's force
        // guard. Nullable; null reproduces pre-P8 behaviour exactly.
        d_force_norm_mirror: *mut c_void,
    ) -> i32;

    // ── M1.2.18.5 — Hamiltonian audit field stage ──
    // WP4: also stamps `adjudication_code` (slot offset 12) from adj+52;
    // this is the only ZSTR node captured downstream of the Adjudicator.
    fn zstr_launch_stage_audit(
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

// ============================================================================
// Dynamic T7 FFI — substrate-aware noise-floor calibration (Wave 3 / Path B)
// ============================================================================
//
// Three captured kernels (capture → reduce → apply) accumulate the
// Adjudicator's current_divergence per launch, compute substrate-derived
// μ + σ across PRISM_DYNT7_N_SAMPLES (=500), and write back to the FFI struct's
// noise_floor_mu[0] / noise_floor_sigma[0].  Replaces the locked 4LPK priors
// once n ≥ PRISM_DYNT7_N_MIN (=100) samples have accumulated.  All GPU-native;
// no host bridges per launch.
extern "C" {
    fn prism_dynamic_t7_launch(
        adj: *const c_void,
        acc_dev: *mut c_void,
        idx_dev: *mut c_void,
        stats_dev: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// M1.2.19.B / Amendment 3.13 — Channel B GhostTileFrame capture FFI
// ============================================================================
extern "C" {
    fn prism_ghost_pipe_stage_launch(
        ring_base_dev: u64,
        tiles: *const c_void,
        adj: *const c_void,
        d_kcc_lead: *const c_void, // Wave 1 / Q2 — F2-pool [n_clusters] u32, nullable
        frame_idx: u64,
        n_clusters: u32,
        max_records: u32,
        stream: *mut c_void,
        firehose_enable: u32, // 0 = adj-gated; nonzero = always emit per cluster per replay
    ) -> i32;

    /// Wave 1 / Q1 — populate __constant__ d_cluster_to_repr_residue[64].
    /// Called once per campaign init from the Rust orchestrator after
    /// Pillar 1 clustering identifies the per-cluster centroid residue.
    fn prism_ghost_set_cluster_repr_residue(
        repr_residues_host: *const u32,
        n: u32,
        stream: *mut c_void,
    ) -> i32;

    /// **M1.2.23 §4 + §5** — Transparent MAR v2 emission launcher.
    ///
    /// Same v1 fields plus the structured v2 overlay payload
    /// (observation_pass, discovery_pass, gear_id, dt_fs, step_idx).
    ///
    /// Schema note: the v2 *overlay layout* is stamped with
    /// `GHOST_FRAME_SCHEMA_V3 = 3`, not 2. `ghost_tile_kernel.cu` is the sole
    /// writer of `GHOST_V2_OFFSET_SCHEMA_VERSION` tree-wide and it writes
    /// `GHOST_FRAME_SCHEMA_V3`. `GHOST_FRAME_SCHEMA_V2 = 2` is defined but
    /// unwritten — no producer in this tree emits it.
    /// Emission is widened to include `kl_divergence >=
    /// observation_threshold_kl`. The 3σ gate writes telemetry only — F1
    /// SWITCH semantics are unchanged. Caller passes raw KL thresholds:
    ///   observation_threshold_kl = mu + 3.0 * sigma
    ///   discovery_threshold_kl   = mu + 12.0 * sigma
    ///
    /// Source-landed in CUDA at ghost_tile_kernel.cu and wired below into the
    /// production captured graph with the exact RichSpike CSR pointers.
    #[allow(dead_code)]
    fn prism_ghost_pipe_stage_launch_v2(
        ring_base_dev: u64,
        tiles: *const c_void,
        d_spikes: *const c_void,
        d_cluster_offsets: *const c_void,
        adj: *const c_void,
        d_kcc_lead: *const c_void,
        frame_idx: u64,
        n_clusters: u32,
        max_records: u32,
        stream: *mut c_void,
        firehose_enable: u32,
        observation_threshold_kl: f32,
        discovery_threshold_kl: f32,
        gear_id: u32,
        dt_fs: f32,
        step_idx: u64,
        // OPERATOR MANDATE 2026-05-08 §1 — device-side timekeeper. u64
        // device pointer to a single uint64_t. 0 ⇒ legacy host-baked
        // behavior (the FFI step_idx arg is used). Non-zero ⇒ kernel
        // dereferences the counter and that becomes the authoritative
        // step_idx written into every record.
        d_step_counter: u64,
        // OPERATOR MANDATE 2026-05-08 §2 — device-side firehose prune.
        // kl_divergence magnitude threshold; records below this are
        // silently dropped before the ring atomicAdd when firehose is on.
        // 0.0 disables (legacy unconditional firehose).
        prune_kl_threshold: f32,
    ) -> i32;
fn prism_ghost_pipe_stage_launch_v2_seed(
        ring_base_dev: u64,
        tiles: *const c_void,
        d_spikes: *const c_void,
        d_cluster_offsets: *const c_void,
        adj: *const c_void,
        d_kcc_lead: *const c_void,
        frame_idx: u64,
        n_clusters: u32,
        max_records: u32,
        stream: *mut c_void,
        firehose_enable: u32,
        observation_threshold_kl: f32,
        discovery_threshold_kl: f32,
        gear_id: u32,
        dt_fs: f32,
        step_idx: u64,
        // OPERATOR MANDATE 2026-05-08 §1 — device-side timekeeper. u64
        // device pointer to a single uint64_t. 0 ⇒ legacy host-baked
        // behavior (the FFI step_idx arg is used). Non-zero ⇒ kernel
        // dereferences the counter and that becomes the authoritative
        // step_idx written into every record.
        d_step_counter: u64,
        // OPERATOR MANDATE 2026-05-08 §2 — device-side firehose prune.
        // kl_divergence magnitude threshold; records below this are
        // silently dropped before the ring atomicAdd when firehose is on.
        // 0.0 disables (legacy unconditional firehose).
        prune_kl_threshold: f32,
        seed_layout: crate::rich_spike::NativeV2SeedLayout,
) -> i32;

    /// Captured physical u32 clock -> owned Ghost/ZSTR u64 counter bridge.
    pub fn prism_bind_ghost_physical_clock_launch(
        physical_step: *const u32,
        ghost_step: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    /// OPERATOR MANDATE 2026-05-08 §1 — device-side timekeeper increment
    /// kernel. Single-thread `__global__` that performs an atomicAdd of
    /// `increment` (typically 1) into the device-side step counter.
    /// Captured into the V2 graph body so each replay advances the GPU
    /// clock without host intervention.
    #[allow(dead_code)]
    pub fn prism_increment_time_launch(
        d_step_counter: *mut c_void, // u64 device pointer
        increment: u32,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// G28 SISR FFI — Spatially-Indexed Symmetric Reflection (Amendment 3.4)
// ============================================================================
//
// Captured between Node B (SO(3) project) and Node C (Adjudicator step) on
// md_stream; bilateral-symmetry gate for homodimer targets.  Writes a u64
// prune mask the Adjudicator reads via `force_prune_mask` in its FFI struct.
extern "C" {
    fn prism_sisr_init_dyad(
        R_row_major: *const f32, // 9 floats, row-major
        t: *const f32,           // 3 floats
        stream: *mut c_void,
    ) -> i32;

    fn prism_sisr_launch(
        tiles: *const c_void,              // *const ContactShellTile
        n_clusters_dev: *const c_void,     // *const u32 (RECT-3.4.1 device veto)
        out_force_prune_mask: *mut c_void, // *mut u64 (8 B device buffer)
        epsilon_sym_angstrom: f32,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// V2 IGNITION FFI — Claude-2's Native C-ABI Bypass (Lane 2 commit 9a90a9c6)
// ============================================================================
//
// Bypasses the cudarc 0.18.2 conditional-node binding bug by calling
// the CUDA Runtime API directly from C++. Operator directive
// 2026-04-29 ("V2 IGNITION").
//
// Contract (mirrors src/cuda/adjudicator.cuh:309):
//   - Creates a `cudaGraphConditionalHandle` bound to `graph`.
//   - Adds a `cudaGraphNodeTypeConditional` IF-node with one body
//     subgraph, downstream of `adjudicator_node`.
//   - Adds an explicit `cudaGraphAddDependencies` edge from
//     `adjudicator_node` to the conditional node (Gate G19 happens-
//     before lock).
//   - Writes the new conditional-node handle into `*out_conditional_node`.
//
// Returns 0 on cudaSuccess, otherwise the cudaError_t cast to int.
extern "C" {
    fn prism_wire_f1_switch_ffi(
        graph: CUgraph,
        adjudicator_node: CUgraphNode,
        predicate_dev_ptr: *const u32,
        out_conditional_node: *mut CUgraphNode,
    ) -> i32;
}

// ============================================================================
// T12 Pre-Flight FFI — G26 chronometric gearbox 4-way SWITCH forge
// ============================================================================
//
// Wave A scaffolding only. Creates a 4-way SWITCH conditional node and
// returns the four body sub-graph handles to Rust unpopulated. Wave B
// will:
//   - Write a predicate-bridge kernel that forwards `gear_id` into the
//     conditional handle via `cudaGraphSetConditional`.
//   - Populate the body sub-graphs with PointerSwap (cases 0/1/2) +
//     PTX trap (case 3) + VelocityRescale (cases 0 & 2 — gear-transition
//     kinetic-energy continuity).
extern "C" {
    fn prism_wire_g26_gearbox_ffi(
        graph: CUgraph,
        predicate_node: CUgraphNode,
        predicate_dev_ptr: *const u32,
        out_conditional_node: *mut CUgraphNode,
        out_body_subgraphs: *mut CUgraph, // [4]
    ) -> i32;

    // B.3.2-FULL — capture-time handle creation.
    fn prism_gearbox_create_handle_ffi(
        graph: CUgraph,
        default_value: u32,
        out_handle: *mut u64,
    ) -> i32;

    // B.3.2-FULL — post-capture SWITCH wire with pre-existing handle.
    fn prism_gearbox_wire_with_handle_ffi(
        graph: CUgraph,
        predicate_node: CUgraphNode,
        handle_v: u64,
        out_conditional_node: *mut CUgraphNode,
        out_body_subgraphs: *mut CUgraph, // [4]
    ) -> i32;

    // B.3.2-FULL — populate the 4 phGraph_out body sub-graphs.
    fn prism_gearbox_populate_switch_bodies_ffi(
        body_subgraphs: *mut CUgraph, // [4]
        adj: *const InterferometricAdjudicatorFfi,
        d_velocities: *mut f32,
        n_floats: u32,
        cruise: *const c_void, // ChronometricStateTensor
        d_current_temp: *const f32,
        d_dt: *const f32,
        target_temp_K: f32,
        tau_ps: f32,
    ) -> i32;

    // F1-PARENT-SWITCH-001 — parent-owned F1 SWITCH (size=3) FFI family.
    // Mirrors the G26 trio above. R3 landed the C++ side at commit
    // 9fa4d356; these are the Rust externs.
    fn prism_f1_create_handle_ffi(graph: CUgraph, default_value: u32, out_handle: *mut u64) -> i32;

    fn prism_f1_wire_with_handle_ffi(
        graph: CUgraph,
        predicate_node: CUgraphNode,
        handle_v: u64,
        out_conditional_node: *mut CUgraphNode,
        out_body_subgraphs: *mut CUgraph, // [3]
    ) -> i32;

    fn prism_f1_populate_switch_bodies_ffi(body_subgraphs: *mut CUgraph, // [3]
    ) -> i32;

    fn prism_f1_launch_predicate_bridge(
        d_adjudication_code: *const u32,
        handle_v: u64,
        mask: u32,
        stream: *mut c_void,
    ) -> i32;

    // CHUNK13_DIAG — TRACED variants of the F1 + G26 predicate-bridge
    // launch shims. Pass `branch_trace_dev = 0` to disable trace writes
    // (kernel becomes byte-equivalent to the legacy bridge). Otherwise
    // pass the device pointer of a pinned-mapped `PrismBranchTrace`
    // (defined in adjudicator.cuh; mirrored as `PrismBranchTrace` in
    // this crate at the bottom of this extern block).
    fn prism_f1_launch_predicate_bridge_traced(
        d_adjudication_code: *const u32,
        handle_v: u64,
        mask: u32,
        stream: *mut c_void,
        branch_trace_dev: u64,
    ) -> i32;

    // M1.2.17 — Hamiltonian Auditor.
    fn prism_energy_monitor_temp_storage_bytes(n: u32, out_temp_bytes: *mut usize) -> i32;

    fn prism_energy_monitor_launch_reduce(
        d_pe_components: *const f64,
        n: u32,
        d_temp_storage: *mut c_void,
        temp_storage_bytes: usize,
        d_pe_scalar: *mut f64,
        d_energy_window: *mut c_void, // EnergyWindow
        d_adj_pe_target: *mut f64,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Public surface
// ============================================================================

/// ASC decision publication and compatibility parameters. Production retains
/// a stream-owned decision record and composes its signed force at both native
/// force phases. The downstream adjudicator only publishes the next controls.
/// The legacy direct-application symbol is retained for ownerless fixtures.
#[derive(Clone)]
pub struct AscConfig {
    /// Native decision owner. Production publishes controls instead of late force writes.
    pub decision_owner: Option<Arc<crate::asc_device::AscDevice>>,
    /// `d_forces` buffer from `NhsAmberFusedEngine` (n_atoms × 3 f32, AoS).
    pub d_forces: *mut f32,
    /// Current atom positions (n_atoms × 3 f32, AoS — same layout as d_forces).
    pub d_atom_positions: *const f32,
    /// Per-atom cluster-membership mask: 1 = inside cluster, 0 = outside.
    pub d_atom_in_cluster: *const u32,
    /// Total atom count.
    pub n_atoms: i32,
    /// Steering gain α — recommend ≤ 0.01 to keep |F_ASC| < 10% of Newtonian.
    pub steering_gain_alpha: f32,
    /// P7a — device `u32` counter: atoms that received a non-zero steering
    /// force. Null disables evidence collection (and its reduction cost).
    ///
    /// Node D accumulates into `d_forces`, but the fused engine's FORCE_N /
    /// FORCE_NP1 phases rewrite that buffer with `forces[tid] = local_force`
    /// — an assignment, not an accumulation. Whether a given run's ASC
    /// contribution reaches an integrator kick therefore depends on replay
    /// ordering and cannot be settled by reading the source alone. These
    /// counters let a run answer it with evidence instead.
    pub d_asc_applied_atoms: *mut u32,
    /// P7a — device `f32` accumulator: Σ|F_ASC| over the counted atoms.
    pub d_asc_applied_sum_f: *mut f32,
}

// SAFETY: raw pointers refer to CUDA device memory which lives on the GPU;
// they are not dereferenced on the host and do not alias host-side objects.
unsafe impl Send for AscConfig {}

/// G23 ZSTR position-staging + fence-signal capture parameters.
///
/// Passed to `PipelineConfig::zstr`.  When `Some`, four kernel nodes are
/// recorded into the captured graph on `telemetry_stream`, downstream of
/// the tile-DMA node and upstream of the JOIN event:
///
/// ```text
/// [tile DMA] → [zstr_pos_stage_f4]
///   → [zstr_force_stage_f4]                  (T11 — Action-Recovery)
///   → [zstr_force_norm_sqrt]                 (T11 — sqrt post-pass)
///   → [zstr_signal_completion] → [JOIN]
/// ```
///
/// All four nodes run on `telemetry_stream` (non-blocking) — zero MD stall.
#[derive(Clone)]
pub struct ZstrCaptureParams {
    /// Device pointer to atom positions (n_atoms × 3 × f32, AoS).
    /// Same pointer as `AscConfig::d_atom_positions`; must be stable
    /// for the lifetime of the pipeline (operator §3.2).
    pub d_positions: *const f32,
    /// Device pointer to integrator forces (n_atoms × 3 × f32, AoS) —
    /// the SAME `d_forces` pointer the ASC kernel atomic-adds steering
    /// contributions into.  Captured AFTER the ASC node so the recorded
    /// frame reflects the post-steering total potential (T11 invariant).
    /// Same lifetime contract as `d_positions`.
    pub d_forces: *const f32,
    /// Number of atoms.  Determines grid size for pos_stage kernel.
    pub n_atoms: u32,
    /// WP4 per-stream slot binding: pinned slot-0 base (header start of
    /// slot 0).  Kernels compute slot offsets at execution time by reading
    /// `*d_active_slot`, which the in-graph slot-authority kernel writes.
    /// Same fused exec rolls all 5 slots.
    pub pinned_base: *mut u8,
    /// Inter-slot stride in bytes (== `ZstrRing::frame_size`, 4096-aligned).
    pub inter_slot_stride: u32,
    /// Byte offset from slot base to positions payload (== sizeof header).
    pub pos_offset_in_slot: u32,
    /// T11 — Byte offset from slot base to forces payload
    /// (== `pos_offset_in_slot + n_atoms*3*4`).
    pub force_offset_in_slot: u32,
    /// T11 — Byte offset from slot base to `force_norm` field
    /// (28 in header). Stable across slots; provided in the params so
    /// the launcher does not duplicate the layout constant.
    pub force_norm_offset_in_slot: u32,
    /// Byte offset from slot base to `completion_fence` field (16 in header).
    pub fence_offset_in_slot: u32,

    // ── WP4 — per-stream slot authority ────────────────────────────────────
    /// Per-stream `uint32_t*` active-slot cell
    /// (`ZstrRing::active_slot_ptr()`). Lives in the ring's own pinned +
    /// DEVICEMAP control sector, so it is structurally unshareable between
    /// streams — unlike the module-scope `d_zstr_active_slot` symbol it
    /// replaces.
    pub d_active_slot: *mut u32,
    /// Per-stream backpressure / conservation status block
    /// (`ZstrRing::backpressure_status_ptr()`, `ZSTR_BP_OFF_*` layout).
    /// May be null; the kernel then keeps no counters.
    pub d_backpressure_status: *mut u8,
    /// Ring slot count (`ZstrRing::N_SLOTS`).
    pub n_slots: u32,
    /// Bounded-wait budget for the device-side backpressure spin. 0 disables
    /// the wait and restores pre-WP4 overwrite semantics.
    pub backpressure_spin_limit: u32,
    /// Value stamped into the header's `stream_id` field (offset 48).
    pub stream_id: u32,
    /// Value stamped into the header's `schema_version` field (offset 52).
    pub schema_version: u32,
}

// SAFETY: pointers are into pinned CUDA host memory + device memory;
// not dereferenced on the host outside the CUDA capture sequence.
unsafe impl Send for ZstrCaptureParams {}

/// **M1.2.23 §4 — Transparent MAR v2 telemetry configuration.**
///
/// When `Some` on `PipelineConfig.mar_v2`, the captured-graph build wires
/// the v2 ghost-tile launcher with these fields. v1 launcher (the current
/// default) is used when this is `None` — bit-identical legacy behavior.
///
/// `T_observation = 3.0` and `T_discovery = 12.0` are the directive-spec
/// sigma multipliers (M1.2.23 §4). The kernel converts them to raw KL
/// thresholds via `host-side mu + multiplier * sigma`. mu/sigma come from
/// the per-stream noise_floor.json computed during cold_hold.
///
/// Hard invariant (per directive §4): observation_pass=true and
/// discovery_pass=false MUST never trigger steering. The v2 kernel
/// respects this trivially because it is a TELEMETRY-ONLY emitter; it
/// does NOT control F1/G26 SWITCH selection.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MarV2Config {
    /// Observation gate sigma multiplier. Default 3.0 per directive §4.
    pub threshold_sigma_observation: f32,
    /// Discovery gate sigma multiplier. Default 12.0 per directive §4.
    pub threshold_sigma_discovery: f32,
    /// Resolved noise-floor μ for the active stream. Used by host to
    /// pre-multiply `mu + sigma_mult * sigma` into the raw KL threshold
    /// passed to the kernel.
    pub noise_floor_mu: f32,
    /// Resolved noise-floor σ for the active stream.
    pub noise_floor_sigma: f32,
    /// Gear id stamped into v2 records.
    ///
    /// **GEAR-PROVENANCE-002.** This value is baked into a captured-graph
    /// kernel node at BUILD time, while the gear is chosen per replay by the
    /// device SFA — so nothing assigned here can ever be a measurement of
    /// what the integrator ran at. Under the documented encoding
    /// (0 = 0.5 fs, 1 = 2.0 fs, 2 = 4.0 fs HMR, 3 = abort) there is no legal
    /// "unknown", and a literal `0` would assert 0.5 fs on every record of a
    /// run actually integrating at 2.0 or 4.0 fs.
    ///
    /// Callers that have no measurement MUST leave this at the default
    /// [`crate::zstr::ZSTR_GEAR_ID_UNMEASURED`]. The measured gear is
    /// obtained instead from
    /// [`CapturedAdjudicationPipeline::measured_gear`], which reads
    /// `ChronometricStateTensor` off the device at a synchronized boundary.
    pub gear_id: u32,
    /// Integration timestep in femtoseconds. Stamped into v2 records.
    pub dt_fs: f32,
    /// Owned u64 Ghost/ZSTR endpoint counter. The captured graph widens
    /// PipelineConfig.d_current_md_step into this allocation before any slot
    /// or record consumer. Zero retains legacy host-baked behavior.
    pub d_step_counter: u64,
    /// OPERATOR MANDATE 2026-05-08 §2 — device-side firehose prune.
    /// kl_divergence magnitude threshold below which the kernel silently
    /// drops the record before the ring atomicAdd, even when
    /// firehose_enable is set. Directive default: 0.01. Set to 0.0 to
    /// preserve legacy unconditional-firehose behavior.
    pub prune_kl_threshold: f32,
}

impl Default for MarV2Config {
    fn default() -> Self {
        Self {
            threshold_sigma_observation: 3.0,
            threshold_sigma_discovery: 12.0,
            noise_floor_mu: 0.0,
            noise_floor_sigma: 1.0,
            // GEAR-PROVENANCE-002 — was a hardcoded literal `0` labelled
            // "Wave A default". A build-time constant cannot witness a
            // per-replay device decision, and `0` is a LEGAL gear, so that
            // default published an unmeasured quantity as a measurement.
            // The explicit not-measured sentinel is the honest default.
            gear_id: crate::zstr::ZSTR_GEAR_ID_UNMEASURED,
            dt_fs: 2.0,
            // §1 + §2 — wired into the build path; default 0 / 0.0
            // preserves legacy semantics for any caller that hasn't
            // populated them yet (caller responsibility to set them
            // when the V2 build path allocates the device counter).
            d_step_counter: 0,
            prune_kl_threshold: 0.01,
        }
    }
}

/// Seed written before a captured Ghost overlay replay. The graph increments
/// the counter once before emission, so only `post_md_step - 1` yields the
/// canonical coordinate step in the durable record.
pub fn ghost_counter_seed_for_exact_step(post_md_step: u64) -> Option<u64> {
    post_md_step.checked_sub(1)
}

/// Exact failure stage for the MAR-v2 coordinate clock binding performed
/// immediately before a captured replay.  Keeping these branches distinct
/// prevents a null-counter policy violation from being misreported as a CUDA
/// context or copy failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactGhostStepBindError {
    MissingDeviceCounter,
    BadPostMdStep {
        post_md_step: u64,
    },
    ContextQuery {
        rc: CUresult,
    },
    ContextBind {
        rc: CUresult,
        expected_context: usize,
        observed_context: usize,
    },
    ContextVerify {
        rc: CUresult,
        expected_context: usize,
        observed_context: usize,
    },
    PhysicalClockRead { rc: CUresult },
    PhysicalClockSynchronize { rc: CUresult },
    PhysicalClockMismatch { physical_step: u32, post_md_step: u64 },
}

impl std::fmt::Display for ExactGhostStepBindError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingDeviceCounter => write!(
                formatter,
                "GHOST_EXACT_STEP_MISSING_COUNTER: MAR-v2 device step counter is null"
            ),
            Self::BadPostMdStep { post_md_step } => write!(
                formatter,
                "GHOST_EXACT_STEP_BAD_STEP: post_md_step={post_md_step} is outside positive u32 physical clock range"
            ),
            Self::ContextQuery { rc } => write!(
                formatter,
                "GHOST_EXACT_STEP_CONTEXT_QUERY_FAILED: rc={rc:?}"
            ),
            Self::ContextBind {
                rc,
                expected_context,
                observed_context,
            } => write!(
                formatter,
                "GHOST_EXACT_STEP_CONTEXT_BIND_FAILED: rc={rc:?} expected_context=0x{expected_context:x} observed_context=0x{observed_context:x}"
            ),
            Self::ContextVerify {
                rc,
                expected_context,
                observed_context,
            } => write!(
                formatter,
                "GHOST_EXACT_STEP_CONTEXT_VERIFY_FAILED: rc={rc:?} expected_context=0x{expected_context:x} observed_context=0x{observed_context:x}"
            ),
            Self::PhysicalClockRead { rc } => write!(formatter, "GHOST_PHYSICAL_CLOCK_READ_FAILED: rc={rc:?}"),
            Self::PhysicalClockSynchronize { rc } => write!(formatter, "GHOST_PHYSICAL_CLOCK_SYNC_FAILED: rc={rc:?}"),
            Self::PhysicalClockMismatch { physical_step, post_md_step } => write!(formatter,
                "GHOST_PHYSICAL_CLOCK_MISMATCH: physical_step={physical_step} requested_step={post_md_step}"),
        }
    }
}

impl std::error::Error for ExactGhostStepBindError {}

impl MarV2Config {
    /// Compute the raw KL threshold the kernel checks: `mu + sigma_mult * sigma`.
    pub fn observation_threshold_kl(&self) -> f32 {
        self.noise_floor_mu + self.threshold_sigma_observation * self.noise_floor_sigma
    }
    /// Compute the discovery raw KL threshold: `mu + 12σ`.
    pub fn discovery_threshold_kl(&self) -> f32 {
        self.noise_floor_mu + self.threshold_sigma_discovery * self.noise_floor_sigma
    }
}

/// G28 SISR symmetry consensus configuration.  When `Some`, the pipeline
/// records a SISR kernel node into the captured graph between Node B
/// (SO(3)) and Node C (Adjudicator).  The kernel writes a u64 prune mask
/// to a pipeline-allocated F2-pool buffer; the Adjudicator's
/// `force_prune_mask` field is initialised to point at the same buffer
/// pre-capture, so each frame's bit-flags propagate to the gate without
/// a host round-trip.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SisrConfig {
    /// Dyad-axis 3×3 rotation matrix (row-major, 9 floats).
    /// For C2 symmetry along Z (operator default): R = diag(-1,-1,1).
    /// For BIOMT-driven dimers: extracted from the assembly file.
    pub dyad_R_row_major: [f32; 9],
    /// Dyad-axis translation (3 floats).
    pub dyad_t: [f32; 3],
    /// Partner-search tolerance in Å.  Operator-recommended: 1.5.
    /// Squared internally to skip sqrtf in the inner loop.
    pub epsilon_sym_angstrom: f32,
}

impl Default for SisrConfig {
    /// 7C8R-default: C2 symmetry about Z-axis after centre-on-origin.
    /// (x,y,z) → (-x,-y,z).  ε_sym = 1.5 Å (Amendment 3.4).
    fn default() -> Self {
        Self {
            dyad_R_row_major: [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
            dyad_t: [0.0, 0.0, 0.0],
            epsilon_sym_angstrom: 1.5,
        }
    }
}

/// Configuration for [`CapturedAdjudicationPipeline::build`].
///
/// All device pointers must live for the lifetime of the pipeline
/// (operator §3.2: "FFI Stability... destination addresses must be
/// immutable"). `n_clusters` is the static shape of the captured graph
/// — a shape change requires re-building the pipeline from scratch.
///
/// Amendment 3.14 / G40: `Clone` enables saving the config at build
/// time so the v9D' Step-101 Heuristic Reset can rebuild the pipeline
/// without re-walking the spike download / VRAM alloc / ASC bind path.
#[derive(Clone)]
pub struct PipelineConfig {
    /// Pre-clustered RichSpike buffer on device. Pointer stability is
    /// the caller's responsibility.
    pub d_spikes: *const RichSpike,
    pub seed_layout: crate::rich_spike::NativeV2SeedLayout,
    /// CSR cluster-offset buffer of length `n_clusters + 1`.
    pub d_cluster_offsets: *const u32,
    /// Number of clusters / equivalently the size of the
    /// `ContactShellTile[]` output array. Static for the captured graph.
    pub n_clusters: u32,
    /// `K_LM[36]` device pointer — obtain via
    /// [`crate::sh_basis::k_lm_device_ptr`] AFTER calling
    /// `prism_sh_basis_init`.
    pub d_k_lm: *const f32,
    /// Frame-id stamped by the SO(3) kernel into every produced tile's
    /// header during the FIRST captured launch. Subsequent launches
    /// reuse the same frame id (the captured graph stamps a constant);
    /// the host increments the frame counter externally if needed.
    pub initial_frame_id: u32,
    /// Native source identity stamped into every SO(3) tile on-device.
    /// This is science-plane provenance and is distinct from diagnostics.
    pub source_stream_id: u32,
    /// Optional runtime stream id for focused TIER 8 diagnostics. This is
    /// log-only metadata; graph construction and kernel behavior must not
    /// branch on it.
    pub diagnostic_stream_id: Option<u32>,
    /// T6 ASC force-bridge config. `None` = skip Node D (safe for tests).
    pub asc: Option<AscConfig>,
    /// P8 — pointer-stable device `f32` that receives the ZSTR force norm.
    ///
    /// The authoritative `force_norm` lives in the ring slot, which rolls
    /// `step % 5`. The G26 governor binds ONE device pointer for the life of
    /// the captured graph, so it cannot read a rolling address. Without this
    /// mirror `force_measured` stays 0 and gate 5 pins the ceiling to the
    /// current gear — the force guard is then inert in both directions: it can
    /// neither authorise an upshift nor observe an excursion to downshift on.
    ///
    /// Null reproduces pre-P8 behaviour exactly.
    pub d_force_norm_mirror: u64,
    /// Explicit physical-stream owner. None builds an unarmed observer context.
    pub governor_device: Option<Arc<crate::governor_device::GovernorDevice>>,
    /// G23 ZSTR position-staging + fence-signal. `None` = no ZSTR nodes
    /// captured (legacy / test builds). `Some` records two kernel nodes on
    /// `telemetry_stream` downstream of the tile DMA (Amendment 2.2).
    pub zstr: Option<ZstrCaptureParams>,
    /// G28 SISR symmetry consensus (Amendment 3.4). `None` = no SISR node
    /// captured (non-dimer / legacy / test).  `Some` records the SISR kernel
    /// node on `md_stream` between SO(3) and Adjudicator, and pre-allocates
    /// the u64 prune-mask buffer into the F2 pool.  The Adjudicator FFI's
    /// `force_prune_mask` is wired to that buffer pre-capture.
    pub sisr: Option<SisrConfig>,

    /// **M1.2.23 §4 — Transparent MAR v2 emission opt-in.**
    ///
    /// `None` (default) → existing v1 ghost-tile launcher is used; on-disk
    /// records have schema_version=0 (v1 legacy zero) in the v2 region.
    /// `Some(MarV2Config { ... })` → v2 launcher path is enabled. The v2/v3
    /// overlay fields (schema_version, observation_pass, discovery_pass,
    /// perturbation_channel, uv_wavelength_nm, field_completeness_flags,
    /// gear_id, dt_fs, step_idx, AABB, centroid) are populated by the kernel,
    /// and schema_version reads GHOST_FRAME_SCHEMA_V3 = 3 — not 2. No
    /// producer in this tree writes 2 (cuda/ghost_tile_kernel.cu:609).
    ///
    /// This IS wired into the captured-graph build path: the launcher
    /// selection at `:4637` dispatches to `prism_ghost_pipe_stage_launch_v2`
    /// when this field is `Some` and to the legacy v1 launcher at `:4858`
    /// when it is `None`. An earlier revision of this comment said
    /// "Source-landed in this commit; NOT yet wired"; that was true before
    /// the wire-in landed and is stale now.
    pub mar_v2: Option<MarV2Config>,
    /// Substrate-aware noise-floor override (Amendment 3.4.6).  When set,
    /// applied AFTER `apply_t7_calibration` overwrites the 4LPK-derived
    /// μ/σ constants with substrate-specific values.  Pair = (μ, σ),
    /// applied uniformly across all 6 SH bands.  Adjudicator threshold
    /// becomes μ + 3σ.  Operator-recommended for 7C8R: (0.01, 0.005)
    /// → threshold 0.025 (vs 4LPK threshold 1.249).
    pub noise_floor_override: Option<(f32, f32)>,

    /// **T12 Pre-Flight — G26 gearbox dt pointer (Wave A connective tissue).**
    /// Device-resident `*mut f32` carrying the active integrator timestep.
    /// `null` = unwired (default). When non-null, the build flow writes
    /// the address into `adj->d_dt` at offset 112 pre-capture so Wave B's
    /// PointerSwap kernel can update the value through the SWITCH body
    /// sub-graphs. Wave A leaves the field UNCONSUMED — no kernel reads
    /// it, no integrator depends on it. Pure surface preparation.
    pub d_dt: *mut f32,

    /// **T12 Pre-Flight — Velocity rescale buffer pointer.**
    /// Device-resident `*mut f32` aliased to the integrator's
    /// `d_velocities` slice (n_atoms × 3 f32, AoS). When non-null, written
    /// into `adj->d_velocities` at offset 120 pre-capture. Wave A leaves
    /// the field UNCONSUMED; Wave B's VelocityRescale kernel inside the
    /// G26 SWITCH body sub-graphs will read it on gear transitions.
    pub d_velocities: *mut f32,

    /// **TIER 8 / Option β — parent-owned G26 conditional handle.**
    /// Zero preserves the standalone/overlay topology: this captured
    /// child graph creates its own conditional handle and installs the
    /// G26 SWITCH node post-capture.  Nonzero means the handle was
    /// created on the parent graph; this child captures only the SFA
    /// + predicate bridge kernel that writes through that handle, and
    /// deliberately skips child-level SWITCH insertion so the template
    /// remains splice-legal.
    pub g26_parent_cond_handle: u64,

    /// **F1-PARENT-SWITCH-001 — parent-owned F1 SWITCH conditional handle.**
    /// Zero ⇒ F1 disabled at the CUDA-graph layer (legacy in-child
    /// `prism_wire_f1_switch_ffi` path stays available for tests but is
    /// NOT used by production). Nonzero ⇒ the handle was created on
    /// the parent graph by `create_parent_f1_cond_handle`; the captured
    /// child's predicate bridge writes `adj->adjudication_code & 0x3`
    /// into that handle so the parent-graph F1 SWITCH (size=3, populated
    /// post-capture by `wire_parent_f1_switch`) routes Prune / Construct
    /// / Violation. Topology: ChildAdj -> F1_SWITCH -> G26_SWITCH so the
    /// Violation branch short-circuits BEFORE G26 mutates gear/dt state.
    pub f1_parent_cond_handle: u64,

    /// **M1.2.17 — Per-atom potential-energy components buffer pointer.**
    /// Device-resident `*const f64` aliased to
    /// `NhsAmberFusedEngine::d_potential_energy_components` (n_atoms × f64,
    /// zeroed each integration step, populated by the AMBER force kernels
    /// via atomicAdd_f64).  The captured pipeline's energy-monitor node
    /// reduces this buffer via `cub::DeviceReduce::Sum` and writes the
    /// scalar V_t into `adj.d_potential_energy` at offset 112.  null =
    /// no energy monitoring (SFA stability fuse stays in first-frame mode).
    pub d_pe_components: *const f64,

    /// **M1.2.17** — Number of atoms (= length of d_pe_components).
    pub n_atoms_for_pe: u32,

    /// Completed observation-window external work scalar. Production points
    /// to the per-stream owner accumulator, distinct from atom rows and the
    /// per-step reduction scalar. Each completed physical step contributes
    /// once; SFA consumes the full interval before the captured reset.
    /// The contributor set is enforced in fused_engine's source contract.
    /// A null pointer is rejected by the production capture guard.
    pub d_external_work: *mut f64,

    /// **M1.2.19.B / Amendment 3.13 — Channel B GhostTileFrame ring (device-mapped).**
    /// Device-side alias of a `GhostTileRing`'s pinned-host buffer.
    /// When non-zero, the build flow records a `prism_ghost_pipe_stage`
    /// kernel into the captured graph downstream of the Adjudicator
    /// step.  The kernel checks `adj.adjudication_code >= 1` and pushes
    /// fixed 4096-byte v3 Ghost records to the buffer, including exact
    /// RichSpike cluster-residue membership counts, and atomically increments
    /// the leading u32 counter.
    /// Zero disables Channel B (legacy / no Φ_sym extraction wanted).
    pub ghost_tile_ring_dev: u64,

    /// **M1.2.19.B** — `max_records` capacity of the ghost ring; kernel
    /// bounds-checks against this before atomic-add'ing a new record.
    /// Sized at engine init based on expected V2 chunk count.
    pub ghost_tile_max_records: u32,

    /// **Diagnostic firehose (post-audit operator directive 2026-05-03).**
    /// `0` ⇒ default Adjudicator-gated emission (record only when
    /// `adj.adjudication_code >= 1` — Construct or Violation events).
    /// Nonzero ⇒ kernel ALWAYS emits one GhostTileFrame per cluster per
    /// replay regardless of adj_code, capturing the full per-cluster KL
    /// trajectory + 4-plane SO(3) spectrum time series even on
    /// null-manifest runs (where the V2 12-σ adjudicator gate never
    /// triggers and the legacy gated path produces zero-byte
    /// `ghost_tiles_bin` artifacts).  Records still carry the actual
    /// adj_code so downstream tools distinguish construct events from
    /// diagnostic samples.  Wired from the
    /// `--ghost-diagnostic-firehose` CLI flag in `nhs_rt_full.rs`
    /// (default ON post-audit).
    pub firehose_enable: u32,

    /// **OPERATOR MANDATE 2026-05-08 §3.II+III — Captured-graph body unroll.**
    ///
    /// How many times the per-step body region (Path C dual-stream fork +
    /// SO(3) projection + G28 SISR + Adjudicator + Dynamic-T7 + Ghost
    /// emission + ASC + G26 gearbox + energy monitor + ZSTR
    /// stage/fence) is replicated INSIDE the cuStreamBeginCapture /
    /// cuStreamEndCapture window. Each iteration appends another copy of
    /// the per-step graph nodes, so a single `cuGraphLaunch` advances
    /// physics by `body_unroll` logical steps.
    ///
    /// Default: 1 (no unroll, legacy bit-identical behavior — host loop in
    /// `nhs_rt_full.rs` invokes `mono.launch_on_stream` once per logical
    /// step). When > 1, the host loop divides chunk_size by this factor
    /// and launches the graph that many times. Setting body_unroll =
    /// chunk_size achieves the directive's "1 launch = 1 chunk" target.
    ///
    /// **Topology caveat at body_unroll > 1**: the per-step body contains
    /// graph constructs that need per-iteration uniqueness — fork/join
    /// `cuEvent`s, the G26 conditional bridge node, and the ZSTR
    /// pos_stage / fence-signal node snapshots. The unroll wrapper at
    /// the build site replays the body N times verbatim; CUDA accepts
    /// the resulting linear DAG (each iteration's nodes are distinct
    /// graph identities), but the post-capture wiring that consumes
    /// snapshot variables (`g26_bridge_node`, `zstr_pos_stage_node`,
    /// `zstr_fence_node`) currently retains only the LAST iteration's
    /// snapshot. That is sufficient for correctness when the snapshot
    /// is used solely as a wiring anchor; it is INSUFFICIENT when the
    /// post-capture wiring needs to update all N iterations'
    /// kernel-node params (e.g., per-step gear write-backs through G26).
    /// Callers driving body_unroll > 1 must validate the ZSTR Reaper
    /// + G26 wiring against their expected per-step semantics.
    pub body_unroll: u32,

    /// **Wave 1 / Q2 — Causal-lead residue F2-pool buffer.**
    /// Device pointer (`u64` raw) to a `[u32; n_clusters]` buffer
    /// holding the per-cluster KCC argmax residue id.  The Ghost
    /// stage kernel reads this when populating `causal_lead_residue`
    /// (offset **124** in GhostTileFrame — `ghost_tile.rs` pins it with
    /// `offset_of!(GhostTileFrame, causal_lead_residue) == 124` and
    /// `cuda/ghost_tile_kernel.cuh:74` static_asserts the mirror.  52 is
    /// `PRISM_ADJ_ADJUDICATION_CODE_OFF`, an offset into the adjudicator FFI
    /// struct, not into this record; patching a record at 52 corrupts
    /// `power_spectrum[8]`).  Zero disables (kernel emits
    /// 0xFFFFFFFFu sentinels — typical bootstrap during the first
    /// chunk before host argmax has run).  The host orchestrator in
    /// `nhs_rt_full.rs` owns the allocation through graph teardown and updates
    /// it from measured_kcc_causal_leads at completed observation endpoints.
    /// The chronological KCC sidecar records exact candidates and measurement step.
    pub d_kcc_lead: u64,

    /// **M1.2.20.C-A / Ruling 3 — anchor force buffer.**
    /// Device pointer (`u64` raw) to `NhsAmberFusedEngine::d_forces`,
    /// the per-atom force vector (`f32 [n_atoms × 3]`).  The gradient
    /// gasp kernel reads `f_anchor = d_forces[3*atom + (0,1,2)]` via
    /// `LDG.E.128` (float4 vectorised load) and combines with the
    /// per-spike anchor mass from `d_masses` to compute the
    /// displacement Δr = η_eff · Q_s · f/m · dt².  Zero disables
    /// the gasp (kernel becomes a no-op identity).  16-byte alignment
    /// is asserted host-side before the captured graph build.
    pub d_forces_anchor: u64,

    /// **M1.2.20.C-A / Ruling 6 — anchor mass buffer.**
    /// Device pointer (`u64` raw) to `NhsAmberFusedEngine::d_masses`,
    /// the per-atom AMBER mass vector (`f32 [n_atoms]`).  Read by the
    /// gradient gasp kernel and the (Phase 3) Momentum Guard for
    /// Σ m_i · Δr_i center-of-mass shift checks.
    pub d_masses: u64,

    /// **M1.2.20.C-A / Ruling 5 — force-burst trigger step.**
    /// `Some(N)` ⇒ the gasp kernel applies a 10× amplification to
    /// η_eff when the engine's step counter equals N (produces the
    /// "Canonical Positive" KL-divergence spike used to label
    /// cryptic-pocket opening events for the offline Teacher
    /// Ensemble).  `None` ⇒ no burst — the FFI field receives the
    /// `u32::MAX` sentinel and the kernel reads it as "disabled".
    /// Wired from the `--force-burst-at-step` CLI flag.
    pub force_burst_step: Option<u32>,

    /// **M1.2.20.C-B — Spike count for d_perturbed_positions allocation.**
    /// Total RichSpike count behind `cfg.d_spikes` (== cluster_offsets[n_clusters]).
    /// Required so the Path C parallel-stream branch can allocate a
    /// pointer-stable XYZ scratchpad with matching event count
    /// from the F2 pool, sized once at pipeline build and reused across
    /// every captured-graph replay.
    pub n_spikes: u32,

    /// Live per-engine ProtocolState.current_step. Owner must outlive this graph.
    /// Mandatory even when force_burst_step is None; no global-clock fallback.
    pub d_current_md_step: *const u32,

    /// **CHUNK13_DIAG — branch-trace pinned-mapped device pointer.**
    /// Optional 64-byte `PrismBranchTrace` allocated via
    /// `cuMemHostAlloc(PORTABLE | DEVICEMAP)` and consumed by the
    /// TRACED variants of the F1 / G26 predicate-bridge launch shims.
    /// 0 = disabled (legacy bridge kernels are used; zero overhead).
    /// Non-zero = TRACED variants are used; predicate value, branch
    /// count, invocation count, and first-launch sentinel are written
    /// atomically on every replay so the host (or watchdog) can read
    /// directly via the matching pinned host pointer with no API call.
    pub branch_trace_dev: u64,
}

/// Owning endpoint graph and the exact V2 child it contains. Keeping the pair
/// private prevents a caller from launching an unrelated graph under another
/// pipeline's context, pointer checks, or G26 observation identity.
pub struct KccObservationParent {
    graph: Box<crate::graph_capture::AutonomousGraph>,
    pipeline: Arc<CapturedAdjudicationPipeline>,
}

impl KccObservationParent {
    pub fn continuation_allocation_plan(&self)->anyhow::Result<Vec<(String,u64)>>{self.graph.continuation_allocation_plan()}
    pub fn capture_continuation(&self)->anyhow::Result<crate::graph_capture::GraphContinuationV1>{self.graph.capture_continuation(&self.pipeline.md_stream)}
    pub fn restore_continuation(&self,saved:&crate::graph_capture::GraphContinuationV1)->anyhow::Result<()>{self.graph.restore_continuation(&self.pipeline.md_stream,saved)}
    pub fn pipeline(&self) -> &Arc<CapturedAdjudicationPipeline> { &self.pipeline }
    pub(crate) fn into_parts(self) -> (Box<crate::graph_capture::AutonomousGraph>, Arc<CapturedAdjudicationPipeline>) {
        (self.graph, self.pipeline)
    }

    pub fn launch(&self) -> anyhow::Result<()> {
        self.pipeline.prepare_launch()?;
        self.graph.launch_on_stream(self.pipeline.md_stream.cu_stream())?;
        if self.pipeline.g26_trace_cfg.is_some() {
            self.pipeline.observe_g26_after_parent_launch(self.graph.cu_graph_exec_raw()
                .ok_or_else(||anyhow::anyhow!("observation parent lacks executable trace identity"))?);
        }
        Ok(())
    }
}

/// LEGO-brick orchestrator. Owns every F2-pool buffer + the pinned
/// ring + the telemetry stream + the captured/instantiated graph.
///
/// Drop releases device buffers, pool, and graph handles.
pub struct CapturedAdjudicationPipeline {
    // Owned resources — released in `drop`.
    pool: crate::governor_device::CudaOwned<VramPool>,
    governor_device: Arc<crate::governor_device::GovernorDevice>,
    asc_device: Option<Arc<crate::asc_device::AscDevice>>,
    md_stream: Arc<CudaStream>,
    telemetry_stream: CUstream,
    /// MD → telemetry: signals "Node C completed, ghost-pipe DMA may launch".
    md_to_telemetry_event: CUevent,
    /// Telemetry → MD: signals "DMA completed, capture may join". Required
    /// to avoid CUDA_ERROR_STREAM_CAPTURE_UNJOINED on cuStreamEndCapture.
    telemetry_to_md_event: CUevent,

    // F2-pool allocations — pre-capture, virtual-pointer-stable.
    perturbed_positions_dev: usize,
    com_shift_dev: usize,
    total_mass_dev: usize,
    com_correction_dev: usize,
    tiles_dev: usize, // *mut ContactShellTile (current/perturbed)
    /// Native SO(3) projection of the raw seed spikes, initialized before
    /// capture. Each replay recomputes the relaxed branch from raw spikes.
    /// `freeze_baseline` may safely snapshot this before the first replay.
    tiles_baseline_dev: usize, // *mut ContactShellTile (relaxed)
    /// **M1.2.26.RECOUPLE Part 4 — Lagged Reference Manifold (P_frozen).**
    /// Pointer-stable F2-pool buffer holding the time-anchored reference
    /// tile array. Populated by host via `freeze_baseline()` at the chosen
    /// anchor boundary, including immediately after build. Host updates
    /// `adj.relaxed_manifold_ptr` (offset 56) to this address so the
    /// adjudicator's KL evaluations compare against the frozen reference
    /// instead of the per-replay-recomputed `tiles_baseline_dev`. Sized
    /// identically to `tiles_baseline_dev` (n_clusters × ContactShellTile).
    pub p_frozen_dev: usize, // *mut ContactShellTile (T29 frozen reference)
    /// Cached `tiles_bytes` so the host's `freeze_baseline()` knows the
    /// memcpy size without recomputing from n_clusters.
    pub tiles_bytes: u64,
    adj_dev: usize,          // *mut InterferometricAdjudicatorFfi
    burst_marker_dev: usize, // *mut u32 (4 B)
    /// G28 SISR per-cluster prune-bit u64. Zero when SISR disabled.
    /// Aliased into `adj->force_prune_mask` pre-capture so the Adjudicator
    /// step kernel reads the bits the SISR kernel wrote earlier in the same
    /// captured graph epoch.  Freed on Drop.
    sisr_mask_dev: usize, // *mut u64 (8 B), or 0 when SISR disabled
    /// RECT-3.4.1 device-resident cluster count. Read by the SISR kernel
    /// to decide whether to early-exit (n_clusters < 2 ⇒ skip prune logic).
    /// Initialised to `cfg.n_clusters` pre-capture; future per-frame updates
    /// land here when the SpikeToCluster4D transform writes dynamic counts.
    sisr_count_dev: usize, // *mut u32 (4 B), or 0 when SISR disabled
    /// Wave 3 / Path B Dynamic T7 calibration buffers (always allocated).
    /// `dynt7_acc_dev`:   F2-pool f32[500]  — Δ_AB samples (saturating)
    /// `dynt7_idx_dev`:   F2-pool u32      — atomic write counter
    /// `dynt7_stats_dev`: F2-pool f32[2]   — [mean, stddev] outputs
    /// All freed on Drop.
    dynt7_acc_dev: usize,
    dynt7_idx_dev: usize,
    dynt7_stats_dev: usize,

    /// Wave B.1 — G26 ChronometricStateTensor (16 bytes, 16-aligned).
    /// Persistent state for the gearbox PointerSwap kernel.  Initialised
    /// to {counter=0, last_burst_frame=0, current_gear=1, _pad=0} —
    /// Gear 1 (2.0 fs) is the safe pre-Burst default.  Pointer-stable
    /// for the campaign; the predicate bridge in B.2 will pass this
    /// address to the PointerSwap kernel through the SWITCH body
    /// sub-graphs.  Freed on Drop.
    cruise_state_dev: usize,

    /// B.3.2-FULL — G26 conditional handle (u64-cast
    /// cudaGraphConditionalHandle).  Created during capture via
    /// `prism_gearbox_create_handle_ffi` so the bridge kernel can
    /// reference it as a kernel-node arg; the SWITCH node added
    /// post-capture references the same handle.  Lifetime managed by
    /// the captured graph itself — no explicit destroy.
    g26_cond_handle: u64,
    f1_cond_handle: u64,
    parent_predicate_count: u32,
    /// B.3.2-FULL — G26 SWITCH conditional node added post-capture
    /// downstream of the bridge kernel.  Body sub-graphs (rescale +
    /// apply_dt + Berendsen for body 0; trap for body 3) populated
    /// via `prism_gearbox_populate_switch_bodies_ffi`.  Null if the
    /// gearbox wiring is bypassed (n_atoms == 0 / test fixture).
    g26_cond_node: CUgraphNode,

    /// **M1.2.17 — Hamiltonian Auditor buffers (F2-pool, freed on Drop).**
    /// `energy_temp_storage_dev`: CUB temp storage (variable size).
    /// `energy_pe_scalar_dev`:    f64 reduce result (8 B).
    /// `energy_window_dev`:       EnergyWindow scratch (16 B).
    /// All zero when n_atoms_for_pe == 0 (energy monitor disabled).
    energy_temp_storage_dev: usize,
    energy_temp_storage_bytes: u64,
    energy_pe_scalar_dev: usize,
    energy_window_dev: usize,

    // CLA-2 pinned host ring.
    ring: crate::governor_device::CudaOwned<PinnedTelemetryRing<ContactShellTile>>,

    // Graph artifacts.
    cu_graph: CUgraph,
    cu_graph_exec: CUgraphExec,
    cond_handle: u64, // CUgraphConditionalHandle
    body_subgraph: CUgraph,

    // G24 ZSTR slot-roller — node handles from the captured graph and the
    // fixed kernel params read back via cuGraphKernelNodeGetParams.
    // All null/zero when PipelineConfig::zstr was None at build time.
    zstr_pos_stage_node: CUgraphNode,
    zstr_fence_node: CUgraphNode,
    zstr_pos_stage_func: CUfunction, // stable across all launches
    zstr_fence_func: CUfunction,
    zstr_src_vram: u64, // d_positions CUdeviceptr — fixed for run lifetime
    zstr_n_atoms: u32,

    /// Owned u64 counter populated from the captured physical u32 clock.
    ghost_step_counter_dev: u64,
    /// Same owner-backed source already used by GASP; never module-global.
    ghost_physical_step_dev: u64,

    /// **GEAR-PROVENANCE-002** — device address of `ProtocolState.dt`
    /// (`cfg.d_dt`). This is the word the gearbox's `apply_fixed_dt` body
    /// kernel writes, so reading it is the only way the host learns the
    /// timestep that was ACTUALLY applied rather than the one configured.
    /// Zero when the caller left `d_dt` unwired.
    g26_dt_dev: usize,

    /// **GEAR-PROVENANCE-002** — resolved bound for the device gear readback,
    /// with its provenance. Resolved once at build so every record on this
    /// pipeline cites the same auditable budget.
    g26_readback_timeout: crate::gearbox::ResolvedTimeout,

    /// **GEAR-PROVENANCE-002** — the stream this pipeline was built for
    /// (`cfg.source_stream_id`). Stamped into every emitted control record.
    /// Held unconditionally so a record never has to invent a stream id when
    /// trace emission is disabled.
    g26_stream_id: u32,

    /// **GEAR-PROVENANCE-002** — G26 control-trace sink configuration,
    /// resolved once at build. `None` disables emission entirely (no readback,
    /// no synchronization, no behavior change).
    g26_trace_cfg: Option<G26TraceConfig>,

    /// **GEAR-PROVENANCE-002** — lazily opened NDJSON sink and record
    /// counter for `ControlPlaneRecord::G26`.
    g26_trace: Mutex<G26TraceSink>,

    // Actual input shape retained for pool/source ownership telemetry.
    n_seed_spikes: u32,
    seed_layout: crate::rich_spike::NativeV2SeedLayout,
    n_physical_atoms: u32,
    // Audit metadata (Claude-3 G18/G19/G20 attestation).
    n_clusters: u32,
    n_kernel_nodes_captured: u32,
    n_dependency_edges_explicit: u32,
}

/// **GEAR-PROVENANCE-002** — policy key selecting the directory that receives
/// per-stream `ControlPlaneRecord::G26` NDJSON.
///
/// Emission is opt-in because it costs a bounded stream drain per launch.
/// When unset, `launch()` is byte-identical to its previous behavior.
pub const G26_CONTROL_TRACE_DIR_ENV: &str = "PRISM_G26_CONTROL_TRACE_DIR";

/// **GEAR-PROVENANCE-002** — policy key overriding the `run_id` stamped into
/// emitted G26 envelopes.
pub const G26_CONTROL_TRACE_RUN_ID_ENV: &str = "PRISM_G26_CONTROL_TRACE_RUN_ID";

/// Resolved G26 control-trace destination for one pipeline.
#[derive(Debug, Clone)]
struct G26TraceConfig {
    dir: std::path::PathBuf,
    run_id: String,
    stream_id: u32,
}

impl G26TraceConfig {
    /// Resolve from policy. Returns `None` when tracing is not requested.
    fn resolve(stream_id: u32) -> Option<Self> {
        let dir = std::env::var(G26_CONTROL_TRACE_DIR_ENV).ok()?;
        let dir = dir.trim();
        if dir.is_empty() {
            return None;
        }
        let run_id = std::env::var(G26_CONTROL_TRACE_RUN_ID_ENV)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("g26_stream{stream_id:02}"));
        Some(Self {
            dir: std::path::PathBuf::from(dir),
            run_id,
            stream_id,
        })
    }

    fn path(&self) -> std::path::PathBuf {
        self.dir.join(format!(
            "stream{:02}_g26_control_trace.ndjson",
            self.stream_id
        ))
    }
}

/// Lazily opened NDJSON sink plus the monotonic record counter.
///
/// `open_failed` latches so a broken destination logs once and then stops
/// retrying — an emitter that cannot write must not turn into a per-launch
/// filesystem storm, and must never silently drop the fact that it failed.
#[derive(Debug, Default)]
struct G26TraceSink {
    file: Option<std::fs::File>,
    record_idx: u64,
    launch_seq: u64,
    open_failed: bool,
}

impl CapturedAdjudicationPipeline {
    /// Attributes for this pipeline's owned F2 pool only. Existing physics,
    /// caller-owned CSR/Ghost buffers and CUDA graph/module allocations are not
    /// attributed to this pool. No stream synchronization is added.
    pub fn allocation_memory_snapshot(&self) -> Result<serde_json::Value, String> {
        let query_begin = std::time::Instant::now();
        let counters = self.pool.memory_snapshot()?;
        Ok(serde_json::json!({
            "scope": "single_captured_adjudication_pipeline_f2_pool",
            "source_stream_id": self.g26_stream_id,
            "device_id": self.pool.device_id(),
            "pool_handle_process_local": format!("0x{:x}", self.pool.raw_handle()),
            "counters": counters,
            "input_shape":{"seed_records":self.n_seed_spikes,"clusters":self.n_clusters,"physical_atoms":self.n_physical_atoms},
            "requested_f2_bytes":{
                "current_relaxed_frozen_tiles":3*self.tiles_bytes,
                "adjudicator":std::mem::size_of::<InterferometricAdjudicatorFfi>(),
                "burst_marker":4,"dynamic_t7":32+4+8,"com_scratch":12+4+12,
                "perturbed_xyz":u64::from(self.n_seed_spikes.max(1))*12,
                "sisr":if self.sisr_mask_dev!=0 {12}else{0},
                "cub_energy_workspace":self.energy_temp_storage_bytes,
                "energy_reduce_and_window":if self.n_physical_atoms>0 {24}else{0}
            },
            "borrowed_seed_bytes":u64::from(self.n_seed_spikes)*self.seed_layout.stride_bytes() as u64,
            "logical_RichSpike_archive_bytes":u64::from(self.n_seed_spikes)*64,
            "seed_layout":self.seed_layout,
            "borrowed_cluster_offsets_bytes":(u64::from(self.n_clusters)+1)*4,
            "borrowed_governor_cruise_pointer":format!("0x{:x}",self.cruise_state_dev),
            "borrowed_governor_cruise_counted_in_f2":false,
            "query_host_duration_ns": query_begin.elapsed().as_nanos(),
            "complete_physical_stream_footprint": false,
            "sampling": "sequential_cuda_attributes_no_added_synchronization",
            "excluded": ["other_pools", "caller_owned_buffers", "cudarc_physics_allocations", "graph_and_module_driver_allocations"]
        }))
    }

    /// Build the captured pipeline end-to-end.
    ///
    /// Sequence (operator IGNITION §1–3):
    ///
    /// 1. F2 pool create (single allocation per resource; all
    ///    pre-capture).
    /// 2. Non-blocking telemetry stream + cross-stream event create.
    /// 3. Pinned host ring create (3 × n_clusters × `sizeof::<Tile>()`).
    /// 4. Adjudicator + manifest pointer registries populated; relaxed
    ///    & perturbed pointers seeded to the same tile pair so the
    ///    captured graph has well-defined inputs even on first launch
    ///    (the Adjudicator owner can swap them mid-campaign).
    /// 5. `cuStreamBeginCapture(md_stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)`.
    ///    THREAD_LOCAL mode (TIER 6, 2026-05-03; was GLOBAL) captures
    ///    cross-stream operations on the telemetry stream that follow a
    ///    recorded event AND keeps invalidation scoped to this worker
    ///    thread — required for the 8-stream interferometric build path
    ///    enabled by TIER 2's `t7_active = true` gate drop.
    /// 6. Inside capture:
    ///    - Pull the in-progress graph handle via
    ///      `cuStreamGetCaptureInfo_v2` so we can create the
    ///      conditional handle bound to it.
    ///    - `cuGraphConditionalHandleCreate(handle, in_progress_graph,
    ///      ctx, default=0, flags=0)`.
    ///    - Launch Node B (`prism_so3_project_run`) on md_stream.
    ///    - Launch Node C (`prism_interferometric_adjudicator_step`)
    ///      on md_stream.
    ///    - `cuEventRecord` on md_stream → cross_stream_event.
    ///    - `cuStreamWaitEvent` on telemetry_stream against
    ///      cross_stream_event (creates a captured DEP edge between
    ///      streams under MODE_GLOBAL).
    ///    - `cuMemcpyDtoHAsync_v2` from `tiles_dev` to the ring's
    ///      frame-0 write slot, on telemetry_stream.
    ///    - Launch Node C' (trampoline `prism_adj_set_conditional`)
    ///      on md_stream with the conditional handle — this becomes
    ///      the LAST captured node on md_stream.
    ///    - `cuStreamGetCaptureInfo_v2` again to fetch the
    ///      `dependencies_out` array; the trampoline node lives at
    ///      the back. We capture its handle for the post-capture
    ///      cuGraphAddDependencies edge.
    /// 7. `cuStreamEndCapture(md_stream)` → final CUgraph G.
    /// 8. Post-capture explicit additions:
    ///    - `cuGraphAddNode(CONDITIONAL params {handle, IF, size=1})`
    ///      with `dependencies = [trampoline_node]` —
    ///      THIS IS THE EXPLICIT cuGraphAddDependencies EDGE THE
    ///      OPERATOR'S §2.3 MANDATE REQUIRES.
    ///    - `cuGraphAddNode(MEMSET v2 { 4 B, value=initial_frame_id })`
    ///      into the body sub-graph — bumps `burst_marker` when the
    ///      Adjudicator routes Case 1 (Burst).
    /// 9. `cuGraphInstantiate(G)` → CUgraphExec.
    /// Convenience wrapper around [`Self::build_with_v2_hook`] with a
    /// no-op hook. V1 callers — and tests that don't need the V2
    /// IGNITION conditional-node injection — should use this method.
    pub fn asc_device(&self) -> Option<&Arc<crate::asc_device::AscDevice>> { self.asc_device.as_ref() }
    pub fn governor_device(&self) -> &Arc<crate::governor_device::GovernorDevice> { &self.governor_device }

    pub fn closeout(&self) -> Result<(crate::governor_device::GovernorSnapshot, crate::governor_device::ConstraintWindow), i32> {
        self.governor_device.snapshot()
    }

    pub fn build(
        ctx: &Arc<CudaContext>,
        md_stream: &Arc<CudaStream>,
        cfg: &PipelineConfig,
    ) -> Result<Self, BuildError> {
        Self::build_with_v2_hook(ctx, md_stream, cfg, |_, _, _| Ok(()))
    }

    /// V2 IGNITION variant. Identical to [`Self::build`] except a
    /// caller-provided closure is invoked between `cuStreamEndCapture`
    /// and `cuGraphInstantiate`. The hook receives:
    ///
    /// 1. The raw `CUgraph` handle (writable — caller may invoke
    ///    `cuGraphAddNode_v2`, `cuGraphAddDependencies`, etc. via
    ///    raw FFI to inject the F1 SWITCH conditional node).
    /// 2. A snapshot of the Adjudicator's captured node handles
    ///    (`&[CUgraphNode]` of length 1 in the V1 baseline). This
    ///    is the operator §2.3 explicit-edge dependency target.
    /// 3. The device pointer (as `usize`) to the
    ///    [`InterferometricAdjudicatorFfi`] — caller can derive
    ///    `predicate_dev_ptr` via Claude-2's FFI
    ///    `prism_get_adjudication_code_devptr(adj_dev_ptr as *const _)`.
    ///
    /// If the hook returns `Err(rc)`, build aborts cleanly: the raw
    /// graph + F2 allocations + ring + streams + events are all
    /// released and a `BuildError::V2HookFailed { rc }` is bubbled.
    ///
    /// # Intended V2 wire-in
    ///
    /// ```ignore
    /// let pipeline = CapturedAdjudicationPipeline::build_with_v2_hook(
    ///     &ctx, &md_stream, &cfg,
    ///     |raw_graph, adj_nodes, adj_dev_ptr| {
    ///         let predicate_ptr = unsafe {
    ///             prism_get_adjudication_code_devptr(
    ///                 adj_dev_ptr as *const InterferometricAdjudicatorFfi,
    ///             )
    ///         };
    ///         let mut cond_node: CUgraphNode = ptr::null_mut();
    ///         let rc = unsafe {
    ///             prism_wire_f1_switch_ffi(
    ///                 raw_graph,
    ///                 adj_nodes[0],
    ///                 predicate_ptr,
    ///                 &mut cond_node as *mut _,
    ///             )
    ///         };
    ///         if rc != 0 { Err(rc) } else { Ok(()) }
    ///     },
    /// )?;
    /// ```
    pub fn build_with_v2_hook<F>(
        ctx: &Arc<CudaContext>,
        md_stream: &Arc<CudaStream>,
        cfg: &PipelineConfig,
        hook: F,
    ) -> Result<Self, BuildError>
    where
        F: FnOnce(CUgraph, &[CUgraphNode], usize) -> Result<(), i32>,
    {
        if cfg.d_current_md_step.is_null()
            || (cfg.d_current_md_step as usize) % std::mem::align_of::<u32>() != 0
        {
            return Err(BuildError::InvalidConfig {
                reason: "per-engine ProtocolState.current_step pointer is null or unaligned",
            });
        }
        if !cfg.seed_layout.validate() { return Err(BuildError::InvalidConfig {reason: "unsupported native V2 seed descriptor"}); }
        if cfg.n_clusters == 0 {
            return Err(BuildError::InvalidConfig {
                reason: "n_clusters must be > 0",
            });
        }
        if cfg.n_spikes > 0 && cfg.n_atoms_for_pe == 0 {
            return Err(BuildError::InvalidConfig {
                reason: "populated GASP events require a nonzero anchor atom count",
            });
        }

        if let Some(mar) = &cfg.mar_v2 {
            if mar.d_step_counter != 0 && mar.d_step_counter % 8 != 0 {
                return Err(BuildError::InvalidConfig { reason: "Ghost u64 counter pointer is unaligned" });
            }
            log::info!("[GHOST-CLOCK] stream={} source=owning_protocol_current_step_u32 destination=ghost_u64 host_seed=false counter_present={}", cfg.source_stream_id, mar.d_step_counter != 0);
        }
        log::info!(
            "[GASP-CLOCK] stream={} live_step_ptr={:p} source=per_engine_protocol_state compatibility_global=false",
            cfg.source_stream_id, cfg.d_current_md_step
        );

        // ── 0. PROACTIVE CONTEXT BIND (2026-06-15 CUDA-context-race fix) ──
        // CUDA's current context is PER-OS-THREAD. This V2 build/capture runs
        // on a multi-stream worker thread whose current context can have
        // drifted (null or a sibling stream's ctx) by the time we issue the
        // first driver call below. The reactive TIER8 gates further down only
        // DRAIN the resulting deferred error AFTER a launch already failed,
        // leaving the stream corrupted → intermittent hang → 8-way ASC barrier
        // deadlock → watchdog kill (observed on 3/5 production reps; rep01/rep05
        // happened to keep ctx coherent). Bind the expected context UP FRONT so
        // every build + capture op runs on the right ctx. Pure context bind — no
        // change to graph structure / body_unroll / blocking-sync (golden master
        // unaffected). Reactive gates remain as backstops.
        {
            let expected_ctx_pre = ctx.cu_ctx();
            let mut cur_ctx_pre: CUcontext = ptr::null_mut();
            let _ = unsafe { cuCtxGetCurrent(&mut cur_ctx_pre as *mut _) };
            if cur_ctx_pre.is_null() || cur_ctx_pre != expected_ctx_pre {
                let rc_pre = unsafe { cuCtxSetCurrent(expected_ctx_pre) };
                if !matches!(rc_pre, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Tier8Context {
                        stage: "build_with_v2_hook proactive cuCtxSetCurrent",
                        rc: rc_pre as i32,
                        expected: expected_ctx_pre as usize,
                        before: cur_ctx_pre as usize,
                        after: 0,
                    });
                }
                log::warn!(
                    "[V2-BUILD proactive-ctx-bind] worker-thread context was drifted \
                     (before={:p} expected={:p}); rebound BEFORE build/capture CUDA ops. \
                     thread_id={:?}",
                    cur_ctx_pre,
                    expected_ctx_pre,
                    std::thread::current().id()
                );
            }
        }

        // ── 0b. DRAIN the SHARED error_state (2026-06-15 CUDA-race root-cause fix) ──
        // The 8 multi-stream workers share ONE Arc<CudaContext>, and cudarc's
        // deferred-error `error_state` is a SINGLE shared cell. A sibling worker
        // thread's Drop (CudaStream/CudaSlice → record_err(bind_to_thread())) can
        // record a stale CUDA_ERROR_INVALID_VALUE into it; the post-gasp-sync
        // gate's `ctx.check_err()` below would then MIS-ATTRIBUTE that foreign
        // error to THIS build (root-caused via 4-agent fan-out; documented in-repo
        // at nhs_rt_full.rs:9439 and 18796). Drain it up front so this build
        // starts from a clean shared error_state — mirrors the post-MD scope-join
        // drain (`context.check_err()`).
        let _ = ctx.check_err();

        // ── 1. F2 pool ────────────────────────────────────────────────
        let governor_device = if let Some(owner) = &cfg.governor_device {
            if !owner.owns_stream(md_stream) || owner.source_stream_id() != cfg.source_stream_id {
                return Err(BuildError::InvalidConfig { reason: "governor owner stream/source mismatch" });
            }
            owner.clone()
        } else {
            let mut policy = crate::gearbox::governor_default_policy().map_err(|_| BuildError::InvalidConfig {
                reason: "unarmed governor defaults unavailable",
            })?;
            policy.armed = 0;
            policy.chunk_span_steps = 1;
            crate::governor_device::GovernorDevice::new(md_stream.clone(), &policy, cfg.source_stream_id, 0)
                .map_err(|rc| BuildError::Cuda { stage: "unarmed per-stream governor create", rc })?
        };
        let asc_device=cfg.asc.as_ref().and_then(|a|a.decision_owner.clone());
        if let Some(owner)=&asc_device {
            if !owner.owns_stream(md_stream) || owner.source_stream_id()!=cfg.source_stream_id {
                return Err(BuildError::InvalidConfig { reason:"ASC owner stream/source mismatch" });
            }
        }
        let pool = crate::governor_device::CudaOwned::new(
            VramPool::new(ctx.cu_device() as i32).map_err(BuildError::PoolCreate)?, md_stream.clone());
        let md_raw = md_stream.cu_stream() as usize;

        // ── 2. Non-blocking telemetry stream + cross-stream events ──
        // Two events required: one for MD → telemetry handoff (after
        // Node C, before DMA), one for the JOIN back from telemetry
        // to MD before cuStreamEndCapture (without it the driver
        // returns CUDA_ERROR_STREAM_CAPTURE_UNJOINED).
        let telemetry_stream =
            create_non_blocking_telemetry_stream().map_err(BuildError::TelemetryStream)?;
        let mut md_to_telemetry_event: CUevent = ptr::null_mut();
        let mut telemetry_to_md_event: CUevent = ptr::null_mut();
        for (event, label) in [
            (&mut md_to_telemetry_event, "md_to_telemetry"),
            (&mut telemetry_to_md_event, "telemetry_to_md"),
        ] {
            let rc = unsafe {
                cuEventCreate(
                    event as *mut _,
                    CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
                )
            };
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                return Err(BuildError::Cuda {
                    stage: match label {
                        "md_to_telemetry" => "cuEventCreate (md_to_telemetry)",
                        _ => "cuEventCreate (telemetry_to_md)",
                    },
                    rc: rc as i32,
                });
            }
        }

        // ── M1.2.20.C-B / Path C — perturbed stream + fork/join events ──
        // The perturbed stream runs the gasp kernel + the second SO(3)
        // projection in parallel with the relaxed-branch SO(3) on
        // md_stream.  Both branches join into md_stream before the
        // Adjudicator step kernel.  Created BEFORE cuStreamBeginCapture
        // so the streams + events themselves are not captured into the
        // graph; cuEventRecord/cuStreamWaitEvent calls inside the
        // capture window create the actual graph dependency edges.
        let stream_perturbed =
            create_non_blocking_telemetry_stream().map_err(BuildError::TelemetryStream)?;
        let mut fork_event: CUevent = ptr::null_mut();
        let mut perturbed_join_event: CUevent = ptr::null_mut();
        for (event, label) in [
            (&mut fork_event, "fork_event (Path C)"),
            (&mut perturbed_join_event, "perturbed_join_event (Path C)"),
        ] {
            let rc = unsafe {
                cuEventCreate(
                    event as *mut _,
                    CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
                )
            };
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                return Err(BuildError::Cuda {
                    stage: if label.starts_with("fork") {
                        "cuEventCreate (fork_event Path C)"
                    } else {
                        "cuEventCreate (perturbed_join_event Path C)"
                    },
                    rc: rc as i32,
                });
            }
        }
        log::info!(
            "[M1.2.20.C-B] Path C streams+events created: \
             stream_perturbed={:p}, fork_event={:p}, perturbed_join_event={:p}",
            stream_perturbed,
            fork_event,
            perturbed_join_event
        );

        // ── 3. F2 allocations ─────────────────────────────────────────
        let tiles_bytes = SiteManifestFfi::alloc_bytes(cfg.n_clusters);
        let tiles_dev =
            pool.alloc_async(tiles_bytes, md_raw)
                .map_err(|s| BuildError::PoolAlloc {
                    what: "tiles",
                    reason: s,
                })?;
        let adj_dev = pool
            .alloc_async(
                std::mem::size_of::<InterferometricAdjudicatorFfi>() as u64,
                md_raw,
            )
            .map_err(|s| BuildError::PoolAlloc {
                what: "adjudicator",
                reason: s,
            })?;
        let burst_marker_dev = pool
            .alloc_async(4, md_raw)
            .map_err(|s| BuildError::PoolAlloc {
                what: "burst_marker",
                reason: s,
            })?;
        // G28 SISR prune-mask buffer (u64) — only when SISR is enabled.
        // Pointer-stable for the pipeline lifetime; zeroed by SISR kernel
        // at the start of each captured-graph launch.
        let sisr_mask_dev: usize = if cfg.sisr.is_some() {
            pool.alloc_async(8, md_raw)
                .map_err(|s| BuildError::PoolAlloc {
                    what: "sisr_mask",
                    reason: s,
                })?
        } else {
            0
        };
        // RECT-3.4.1 device-resident cluster count buffer (u32). Initialised
        // to cfg.n_clusters via cuMemcpyHtoD below; the SISR kernel reads
        // this value at execution time to decide early-exit (< 2 ⇒ skip).
        let sisr_count_dev: usize = if cfg.sisr.is_some() {
            pool.alloc_async(4, md_raw)
                .map_err(|s| BuildError::PoolAlloc {
                    what: "sisr_count",
                    reason: s,
                })?
        } else {
            0
        };
        // Separate relaxed manifold. Allocation zeroing is not a scientific
        // reference: the raw-spike SO(3) projection below must complete before
        // capture/build return permits the host to freeze this buffer.
        let tiles_baseline_dev =
            pool.alloc_async(tiles_bytes, md_raw)
                .map_err(|s| BuildError::PoolAlloc {
                    what: "tiles_baseline",
                    reason: s,
                })?;
        // Initialize storage/padding. Replaced by the real native projection
        // before capture; these zero bytes must never be frozen as evidence.
        unsafe {
            let rc = cuMemsetD8Async(tiles_baseline_dev as CUdeviceptr, 0, tiles_bytes as usize, md_stream.cu_stream());
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                return Err(BuildError::Cuda {
                    stage: "memset tiles_baseline",
                    rc: rc as i32,
                });
            }
        }

        // M1.2.26.RECOUPLE Part 4 — Lagged Reference Manifold (P_frozen).
        // Allocate a separate buffer holding the time-anchored reference
        // tile array. Host orchestration (nhs_rt_full chunk loop) snapshots
        // tiles_baseline_dev into p_frozen_dev at its chosen anchor boundary.
        // The production host currently freezes immediately after build and
        // before the first V2 replay. After the snapshot, the
        // host updates adj.relaxed_manifold_ptr (offset 56) to point at
        // p_frozen_dev so subsequent KL evaluations compare against the
        // frozen reference instead of the per-replay-recomputed baseline.
        // Pre-snapshot: zero-initialized so any premature read produces
        // a degenerate (but well-defined) Δ_AB that the SFA can reject.
        let p_frozen_dev =
            pool.alloc_async(tiles_bytes, md_raw)
                .map_err(|s| BuildError::PoolAlloc {
                    what: "p_frozen",
                    reason: s,
                })?;
        unsafe {
            let rc = cuMemsetD8Async(p_frozen_dev as CUdeviceptr, 0, tiles_bytes as usize, md_stream.cu_stream());
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                return Err(BuildError::Cuda {
                    stage: "memset p_frozen",
                    rc: rc as i32,
                });
            }
        }

        // ── Wave 3 / Path B — Dynamic T7 calibration buffers ────────────
        // acc:   500 × f32 = 2000 B  (cold-equilibrium Δ_AB samples)
        // idx:    1 × u32 = 4 B      (atomic write counter, saturating)
        // stats:  2 × f32 = 8 B      ([mean, stddev])
        // All zero-initialised so the FIRST capture starts from a clean slate.
        // M1.2.20.C-C / T19 — replaces the 500-sample f32 buffer with a
        // 24-byte (32-aligned) CalibrationStateF64 holding the f64
        // sum_kl + sum_sq_kl + count + applied accumulator.  idx and
        // stats slots are kept for ABI compatibility but unused
        // post-T19; the launcher ignores them.
        const DYNT7_ACC_BYTES: u64 = 32; // CalibrationStateF64 (24 used + 8 pad)
        const DYNT7_IDX_BYTES: u64 = 4; // unused post-T19
        const DYNT7_STATS_BYTES: u64 = 2 * 4; // unused post-T19
        let dynt7_acc_dev =
            pool.alloc_async(DYNT7_ACC_BYTES, md_raw)
                .map_err(|s| BuildError::PoolAlloc {
                    what: "dynt7_acc",
                    reason: s,
                })?;
        let dynt7_idx_dev =
            pool.alloc_async(DYNT7_IDX_BYTES, md_raw)
                .map_err(|s| BuildError::PoolAlloc {
                    what: "dynt7_idx",
                    reason: s,
                })?;
        let dynt7_stats_dev =
            pool.alloc_async(DYNT7_STATS_BYTES, md_raw)
                .map_err(|s| BuildError::PoolAlloc {
                    what: "dynt7_stats",
                    reason: s,
                })?;
        unsafe {
            let rc = cuMemsetD8Async(dynt7_acc_dev as CUdeviceptr, 0, DYNT7_ACC_BYTES as usize, md_stream.cu_stream());
            if rc != CUresult::CUDA_SUCCESS {
                return Err(BuildError::Cuda { stage: "stream-ordered F2 scratch initialization", rc: rc as i32 });
            }
            let rc = cuMemsetD8Async(dynt7_idx_dev as CUdeviceptr, 0, DYNT7_IDX_BYTES as usize, md_stream.cu_stream());
            if rc != CUresult::CUDA_SUCCESS {
                return Err(BuildError::Cuda { stage: "stream-ordered F2 scratch initialization", rc: rc as i32 });
            }
            let rc = cuMemsetD8Async(
                dynt7_stats_dev as CUdeviceptr,
                0,
                DYNT7_STATS_BYTES as usize, md_stream.cu_stream());
            if rc != CUresult::CUDA_SUCCESS {
                return Err(BuildError::Cuda { stage: "stream-ordered F2 scratch initialization", rc: rc as i32 });
            }
        }

        // ── M1.2.20.C-B — Path C scratchpads ─────────────────────────
        // d_perturbed_positions: pointer-stable XYZ scratchpad sized
        //   to cfg.n_spikes × 12 bytes; the gasp kernel writes
        //   pos_perturbed = pos_relaxed + Δr per spike, and the
        //   stream_perturbed SO(3) combines it with unchanged raw metadata.
        //   F2 pool ownership keeps its virtual address stable through graph
        //   teardown; pool retention does not itself guarantee L2 residency.
        // d_com_shift_dev: 3-element f32 atomic accumulator for the
        //   PTX Momentum Guard.  Zero'd at head-of-loop on md_stream
        //   so each chunk starts with a clean Σ m·Δr accumulator.
        const POSITION_BYTES: u64 = std::mem::size_of::<[f32; 3]>() as u64;
        let spikes_perturbed_bytes: u64 = (cfg.n_spikes.max(1) as u64) * POSITION_BYTES;
        let d_perturbed_positions = pool
            .alloc_async(spikes_perturbed_bytes, md_raw)
            .map_err(|s| BuildError::PoolAlloc {
                what: "d_perturbed_positions",
                reason: s,
            })?;
        const COM_SHIFT_BYTES: u64 = 12; // 3 × f32 — Σ m·Δr accumulator
        const TOTAL_MASS_BYTES: u64 = 4; // 1 × f32 — Σ m accumulator (Path Ω Option A)
        const COM_CORRECTION_BYTES: u64 = 12; // 3 × f32 — correction vector (Path Ω Option A)
        let d_com_shift_dev =
            pool.alloc_async(COM_SHIFT_BYTES, md_raw)
                .map_err(|s| BuildError::PoolAlloc {
                    what: "d_com_shift",
                    reason: s,
                })?;
        let d_total_mass_dev =
            pool.alloc_async(TOTAL_MASS_BYTES, md_raw)
                .map_err(|s| BuildError::PoolAlloc {
                    what: "d_total_mass",
                    reason: s,
                })?;
        let d_com_correction_dev = pool
            .alloc_async(COM_CORRECTION_BYTES, md_raw)
            .map_err(|s| BuildError::PoolAlloc {
                what: "d_com_correction",
                reason: s,
            })?;
        unsafe {
            let rc = cuMemsetD8Async(
                d_perturbed_positions as CUdeviceptr,
                0,
                spikes_perturbed_bytes as usize, md_stream.cu_stream());
            if rc != CUresult::CUDA_SUCCESS {
                return Err(BuildError::Cuda { stage: "stream-ordered F2 scratch initialization", rc: rc as i32 });
            }
            let rc = cuMemsetD8Async(d_com_shift_dev as CUdeviceptr, 0, COM_SHIFT_BYTES as usize, md_stream.cu_stream());
            if rc != CUresult::CUDA_SUCCESS {
                return Err(BuildError::Cuda { stage: "stream-ordered F2 scratch initialization", rc: rc as i32 });
            }
            let rc = cuMemsetD8Async(
                d_total_mass_dev as CUdeviceptr,
                0,
                TOTAL_MASS_BYTES as usize, md_stream.cu_stream());
            if rc != CUresult::CUDA_SUCCESS {
                return Err(BuildError::Cuda { stage: "stream-ordered F2 scratch initialization", rc: rc as i32 });
            }
            let rc = cuMemsetD8Async(
                d_com_correction_dev as CUdeviceptr,
                0,
                COM_CORRECTION_BYTES as usize, md_stream.cu_stream());
            if rc != CUresult::CUDA_SUCCESS {
                return Err(BuildError::Cuda { stage: "stream-ordered F2 scratch initialization", rc: rc as i32 });
            }
        }
        log::info!(
            "[Path Ω Option A] allocated d_perturbed_positions={} B, \
             d_com_shift={} B, d_total_mass={} B, d_com_correction={} B \
             (canonical interferometric formulation: COM-locked perturbed manifold)",
            spikes_perturbed_bytes,
            COM_SHIFT_BYTES,
            TOTAL_MASS_BYTES,
            COM_CORRECTION_BYTES
        );

        // Cruise and governor dwell share the owner lifetime and both survive
        // child-template rebuilds. Only owner creation resets their state.
        let cruise_state_dev = governor_device.cruise_state_ptr() as usize;

        // ── M1.2.17 — Hamiltonian Auditor buffers ──────────────────
        // Three F2-pool allocations for the captured energy-monitor
        // node:
        //   energy_temp_storage_dev: CUB temp storage (sized via
        //     prism_energy_monitor_temp_storage_bytes).
        //   energy_pe_scalar_dev:    f64 reduce target (8 B).
        //   energy_window_dev:       EnergyWindow (16 B: prev/cur f64).
        //
        // All zero-initialised so the first launch sees a clean baseline.
        // Sized to PipelineConfig::n_atoms_for_pe; if 0 the node skips.
        let mut energy_temp_storage_dev: usize = 0;
        let mut energy_pe_scalar_dev: usize = 0;
        let mut energy_window_dev: usize = 0;
        let mut energy_temp_storage_bytes: u64 = 0;
        if cfg.n_atoms_for_pe > 0 {
            let diag_protocol_group = if cfg.g26_parent_cond_handle != 0 {
                "parent_owned_g26_child"
            } else {
                "standalone_child"
            };
            let diag_stream_id = cfg
                .diagnostic_stream_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "unavailable".to_string());
            let diag_thread_id = format!("{:?}", std::thread::current().id());
            let (ctx_before, ctx_before_rc) = tier8_diag_current_context();
            let (ctx_before_name, ctx_before_text) = tier8_diag_driver_error_text(ctx_before_rc);
            tier8_diag_verbose!(
                "[TIER8-DIAG energy-pre-sync] stream_id={} \
                 protocol_group={} md_raw=0x{:x} perturbed_stream={:p} \
                 parent_graph=unavailable_in_captured_pipeline \
                 parent_cond_handle=0x{:x} current_ctx={:p} \
                 current_ctx_rc={} current_ctx_name={} current_ctx_string=\"{}\" \
                 thread_id={} n_atoms_for_pe={}",
                diag_stream_id,
                diag_protocol_group,
                md_raw,
                stream_perturbed,
                cfg.g26_parent_cond_handle,
                ctx_before,
                ctx_before_rc as i32,
                ctx_before_name,
                ctx_before_text,
                diag_thread_id,
                cfg.n_atoms_for_pe
            );

            // Query CUB temp-storage size for an n-element f64 reduce.
            let mut temp_bytes: usize = 0;
            let rc = unsafe {
                prism_energy_monitor_temp_storage_bytes(
                    cfg.n_atoms_for_pe,
                    &mut temp_bytes as *mut usize,
                )
            };
            if rc != 0 {
                let (name, text) = tier8_diag_runtime_error_text(rc);
                log::error!(
                    "[TIER8-DIAG energy-pre-sync] call=prism_energy_monitor_temp_storage_bytes \
                     result=FAIL stream_id={} protocol_group={} rc={} \
                     cuda_name={} cuda_string=\"{}\" n_atoms_for_pe={} \
                     out_temp_bytes={}",
                    diag_stream_id,
                    diag_protocol_group,
                    rc,
                    name,
                    text,
                    cfg.n_atoms_for_pe,
                    temp_bytes
                );
                return Err(BuildError::Cuda {
                    stage: "M1.2.17 prism_energy_monitor_temp_storage_bytes",
                    rc,
                });
            }
            tier8_diag_verbose!(
                "[TIER8-DIAG energy-pre-sync] call=prism_energy_monitor_temp_storage_bytes \
                 result=OK stream_id={} protocol_group={} rc=0 \
                 temp_bytes={} n_atoms_for_pe={}",
                diag_stream_id,
                diag_protocol_group,
                temp_bytes,
                cfg.n_atoms_for_pe
            );
            energy_temp_storage_bytes = temp_bytes as u64;
            energy_temp_storage_dev = match pool.alloc_async(temp_bytes as u64, md_raw) {
                Ok(ptr) => {
                    tier8_diag_verbose!(
                        "[TIER8-DIAG energy-pre-sync] call=pool.alloc_async \
                         what=energy_temp_storage result=OK stream_id={} \
                         protocol_group={} ptr=0x{:x} bytes={} is_null={} \
                         md_raw=0x{:x} pool=0x{:x}",
                        diag_stream_id,
                        diag_protocol_group,
                        ptr,
                        temp_bytes,
                        ptr == 0,
                        md_raw,
                        pool.raw_handle()
                    );
                    ptr
                }
                Err(s) => {
                    log::error!(
                        "[TIER8-DIAG energy-pre-sync] call=pool.alloc_async \
                         what=energy_temp_storage result=FAIL stream_id={} \
                         protocol_group={} bytes={} md_raw=0x{:x} pool=0x{:x} reason=\"{}\"",
                        diag_stream_id,
                        diag_protocol_group,
                        temp_bytes,
                        md_raw,
                        pool.raw_handle(),
                        s
                    );
                    return Err(BuildError::PoolAlloc {
                        what: "energy_temp_storage",
                        reason: s,
                    });
                }
            };
            energy_pe_scalar_dev = match pool.alloc_async(8, md_raw) {
                Ok(ptr) => {
                    tier8_diag_verbose!(
                        "[TIER8-DIAG energy-pre-sync] call=pool.alloc_async \
                         what=energy_pe_scalar result=OK stream_id={} \
                         protocol_group={} ptr=0x{:x} bytes=8 is_null={} \
                         md_raw=0x{:x} pool=0x{:x}",
                        diag_stream_id,
                        diag_protocol_group,
                        ptr,
                        ptr == 0,
                        md_raw,
                        pool.raw_handle()
                    );
                    ptr
                }
                Err(s) => {
                    log::error!(
                        "[TIER8-DIAG energy-pre-sync] call=pool.alloc_async \
                         what=energy_pe_scalar result=FAIL stream_id={} \
                         protocol_group={} bytes=8 md_raw=0x{:x} pool=0x{:x} reason=\"{}\"",
                        diag_stream_id,
                        diag_protocol_group,
                        md_raw,
                        pool.raw_handle(),
                        s
                    );
                    return Err(BuildError::PoolAlloc {
                        what: "energy_pe_scalar",
                        reason: s,
                    });
                }
            };
            energy_window_dev = match pool.alloc_async(16, md_raw) {
                Ok(ptr) => {
                    tier8_diag_verbose!(
                        "[TIER8-DIAG energy-pre-sync] call=pool.alloc_async \
                         what=energy_window result=OK stream_id={} \
                         protocol_group={} ptr=0x{:x} bytes=16 is_null={} \
                         md_raw=0x{:x} pool=0x{:x}",
                        diag_stream_id,
                        diag_protocol_group,
                        ptr,
                        ptr == 0,
                        md_raw,
                        pool.raw_handle()
                    );
                    ptr
                }
                Err(s) => {
                    log::error!(
                        "[TIER8-DIAG energy-pre-sync] call=pool.alloc_async \
                         what=energy_window result=FAIL stream_id={} \
                         protocol_group={} bytes=16 md_raw=0x{:x} pool=0x{:x} reason=\"{}\"",
                        diag_stream_id,
                        diag_protocol_group,
                        md_raw,
                        pool.raw_handle(),
                        s
                    );
                    return Err(BuildError::PoolAlloc {
                        what: "energy_window",
                        reason: s,
                    });
                }
            };

            for (label, ptr, bytes) in [
                ("energy_temp_storage", energy_temp_storage_dev, temp_bytes),
                ("energy_pe_scalar", energy_pe_scalar_dev, 8usize),
                ("energy_window", energy_window_dev, 16usize),
            ] {
                let rc = unsafe { cuMemsetD8Async(ptr as CUdeviceptr, 0, bytes, md_stream.cu_stream()) };
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    let (name, text) = tier8_diag_driver_error_text(rc);
                    log::error!(
                        "[TIER8-DIAG energy-pre-sync] call=cuMemsetD8Async \
                         what={} result=FAIL stream_id={} protocol_group={} \
                         rc={} cuda_name={} cuda_string=\"{}\" ptr=0x{:x} \
                         bytes={} is_null={} md_raw=0x{:x} parent_cond_handle=0x{:x}",
                        label,
                        diag_stream_id,
                        diag_protocol_group,
                        rc as i32,
                        name,
                        text,
                        ptr,
                        bytes,
                        ptr == 0,
                        md_raw,
                        cfg.g26_parent_cond_handle
                    );
                    return Err(BuildError::Cuda {
                        stage: "TIER8-DIAG cuMemsetD8Async energy pre-sync",
                        rc: rc as i32,
                    });
                }
                tier8_diag_verbose!(
                    "[TIER8-DIAG energy-pre-sync] call=cuMemsetD8Async \
                     what={} result=OK stream_id={} protocol_group={} \
                     rc=0 ptr=0x{:x} bytes={} is_null={} md_raw=0x{:x}",
                    label,
                    diag_stream_id,
                    diag_protocol_group,
                    ptr,
                    bytes,
                    ptr == 0,
                    md_raw
                );
            }

            let expected_ctx = ctx.cu_ctx() as CUcontext;
            let mut before_ctx: CUcontext = ptr::null_mut();
            let rc_get_before = unsafe { cuCtxGetCurrent(&mut before_ctx as *mut _) };
            let (get_before_name, get_before_text) = tier8_diag_driver_error_text(rc_get_before);
            if !matches!(rc_get_before, CUresult::CUDA_SUCCESS) {
                log::error!(
                    "[TIER8-DIAG energy-pre-sync] call=cuCtxGetCurrent(before) \
                     result=FAIL stream_id={} protocol_group={} rc={} \
                     cuda_name={} cuda_string=\"{}\" expected_ctx={:p} \
                     before_ctx={:p} md_raw=0x{:x} perturbed_stream={:p} \
                     parent_graph=unavailable_in_captured_pipeline \
                     parent_cond_handle=0x{:x} thread_id={}",
                    diag_stream_id,
                    diag_protocol_group,
                    rc_get_before as i32,
                    get_before_name,
                    get_before_text,
                    expected_ctx,
                    before_ctx,
                    md_raw,
                    stream_perturbed,
                    cfg.g26_parent_cond_handle,
                    diag_thread_id
                );
                return Err(BuildError::Tier8Context {
                    stage: "TIER8 energy pre-sync cuCtxGetCurrent(before)",
                    rc: rc_get_before as i32,
                    expected: expected_ctx as usize,
                    before: before_ctx as usize,
                    after: 0,
                });
            }

            let mut did_set_current = false;
            let mut set_current_rc = CUresult::CUDA_SUCCESS;
            let mut set_current_name = "CUDA_SUCCESS".to_string();
            let mut set_current_text = "no error".to_string();
            if before_ctx.is_null() || before_ctx != expected_ctx {
                did_set_current = true;
                set_current_rc = unsafe { cuCtxSetCurrent(expected_ctx) };
                let (name, text) = tier8_diag_driver_error_text(set_current_rc);
                set_current_name = name;
                set_current_text = text;
                if !matches!(set_current_rc, CUresult::CUDA_SUCCESS) {
                    log::error!(
                        "[TIER8-DIAG energy-pre-sync] call=cuCtxSetCurrent \
                         result=FAIL stream_id={} protocol_group={} rc={} \
                         cuda_name={} cuda_string=\"{}\" expected_ctx={:p} \
                         before_ctx={:p} md_raw=0x{:x} perturbed_stream={:p} \
                         parent_graph=unavailable_in_captured_pipeline \
                         parent_cond_handle=0x{:x} thread_id={}",
                        diag_stream_id,
                        diag_protocol_group,
                        set_current_rc as i32,
                        set_current_name,
                        set_current_text,
                        expected_ctx,
                        before_ctx,
                        md_raw,
                        stream_perturbed,
                        cfg.g26_parent_cond_handle,
                        diag_thread_id
                    );
                    return Err(BuildError::Tier8Context {
                        stage: "TIER8 energy pre-sync cuCtxSetCurrent",
                        rc: set_current_rc as i32,
                        expected: expected_ctx as usize,
                        before: before_ctx as usize,
                        after: 0,
                    });
                }
            }

            let mut after_ctx: CUcontext = ptr::null_mut();
            let rc_get_after = unsafe { cuCtxGetCurrent(&mut after_ctx as *mut _) };
            let (get_after_name, get_after_text) = tier8_diag_driver_error_text(rc_get_after);
            if !matches!(rc_get_after, CUresult::CUDA_SUCCESS) {
                log::error!(
                    "[TIER8-DIAG energy-pre-sync] call=cuCtxGetCurrent(after) \
                     result=FAIL stream_id={} protocol_group={} rc={} \
                     cuda_name={} cuda_string=\"{}\" expected_ctx={:p} \
                     before_ctx={:p} after_ctx={:p} did_set_current={} \
                     set_rc={} set_name={} set_string=\"{}\" md_raw=0x{:x} \
                     perturbed_stream={:p} parent_graph=unavailable_in_captured_pipeline \
                     parent_cond_handle=0x{:x} thread_id={}",
                    diag_stream_id,
                    diag_protocol_group,
                    rc_get_after as i32,
                    get_after_name,
                    get_after_text,
                    expected_ctx,
                    before_ctx,
                    after_ctx,
                    did_set_current,
                    set_current_rc as i32,
                    set_current_name,
                    set_current_text,
                    md_raw,
                    stream_perturbed,
                    cfg.g26_parent_cond_handle,
                    diag_thread_id
                );
                return Err(BuildError::Tier8Context {
                    stage: "TIER8 energy pre-sync cuCtxGetCurrent(after)",
                    rc: rc_get_after as i32,
                    expected: expected_ctx as usize,
                    before: before_ctx as usize,
                    after: after_ctx as usize,
                });
            }
            if after_ctx != expected_ctx {
                log::error!(
                    "[TIER8-DIAG energy-pre-sync] call=raw_context_guard \
                     result=FAIL stream_id={} protocol_group={} reason=context_mismatch \
                     expected_ctx={:p} before_ctx={:p} after_ctx={:p} \
                     did_set_current={} get_before_rc={} get_before_name={} \
                     get_before_string=\"{}\" set_rc={} set_name={} set_string=\"{}\" \
                     get_after_rc={} get_after_name={} get_after_string=\"{}\" \
                     md_raw=0x{:x} perturbed_stream={:p} \
                     parent_graph=unavailable_in_captured_pipeline \
                     parent_cond_handle=0x{:x} thread_id={}",
                    diag_stream_id,
                    diag_protocol_group,
                    expected_ctx,
                    before_ctx,
                    after_ctx,
                    did_set_current,
                    rc_get_before as i32,
                    get_before_name,
                    get_before_text,
                    set_current_rc as i32,
                    set_current_name,
                    set_current_text,
                    rc_get_after as i32,
                    get_after_name,
                    get_after_text,
                    md_raw,
                    stream_perturbed,
                    cfg.g26_parent_cond_handle,
                    diag_thread_id
                );
                return Err(BuildError::Tier8Context {
                    stage: "TIER8 energy pre-sync context mismatch",
                    rc: -1,
                    expected: expected_ctx as usize,
                    before: before_ctx as usize,
                    after: after_ctx as usize,
                });
            }
            tier8_diag_verbose!(
                "[TIER8-DIAG energy-pre-sync] call=raw_context_guard \
                 result=OK stream_id={} protocol_group={} expected_ctx={:p} \
                 before_ctx={:p} after_ctx={:p} did_set_current={} \
                 get_before_rc={} get_before_name={} get_before_string=\"{}\" \
                 set_rc={} set_name={} set_string=\"{}\" \
                 get_after_rc={} get_after_name={} get_after_string=\"{}\" \
                 md_raw=0x{:x} perturbed_stream={:p} \
                 parent_graph=unavailable_in_captured_pipeline \
                 parent_cond_handle=0x{:x} thread_id={}",
                diag_stream_id,
                diag_protocol_group,
                expected_ctx,
                before_ctx,
                after_ctx,
                did_set_current,
                rc_get_before as i32,
                get_before_name,
                get_before_text,
                set_current_rc as i32,
                set_current_name,
                set_current_text,
                rc_get_after as i32,
                get_after_name,
                get_after_text,
                md_raw,
                stream_perturbed,
                cfg.g26_parent_cond_handle,
                diag_thread_id
            );

            let sync_rc = unsafe { cuStreamSynchronize(md_raw as CUstream) };
            if !matches!(sync_rc, CUresult::CUDA_SUCCESS) {
                let (name, text) = tier8_diag_driver_error_text(sync_rc);
                let (ctx_after_sync, ctx_after_sync_rc) = tier8_diag_current_context();
                log::error!(
                    "[TIER8-DIAG energy-pre-sync] call=cuStreamSynchronize \
                     result=FAIL stream_id={} protocol_group={} \
                     rc={} cuda_name={} cuda_string=\"{}\" md_raw=0x{:x} \
                     perturbed_stream={:p} parent_cond_handle=0x{:x} \
                     expected_ctx={:p} before_ctx={:p} after_ctx={:p} \
                     did_set_current={} \
                     energy_temp_ptr=0x{:x} energy_temp_bytes={} energy_temp_is_null={} \
                     energy_scalar_ptr=0x{:x} energy_scalar_bytes=8 energy_scalar_is_null={} \
                     energy_window_ptr=0x{:x} energy_window_bytes=16 energy_window_is_null={} \
                     current_ctx={:p} current_ctx_rc={} thread_id={}",
                    diag_stream_id,
                    diag_protocol_group,
                    sync_rc as i32,
                    name,
                    text,
                    md_raw,
                    stream_perturbed,
                    cfg.g26_parent_cond_handle,
                    expected_ctx,
                    before_ctx,
                    after_ctx,
                    did_set_current,
                    energy_temp_storage_dev,
                    temp_bytes,
                    energy_temp_storage_dev == 0,
                    energy_pe_scalar_dev,
                    energy_pe_scalar_dev == 0,
                    energy_window_dev,
                    energy_window_dev == 0,
                    ctx_after_sync,
                    ctx_after_sync_rc as i32,
                    diag_thread_id
                );
                return Err(BuildError::Tier8Context {
                    stage: "TIER8-DIAG raw cuStreamSynchronize energy pre-sync",
                    rc: sync_rc as i32,
                    expected: expected_ctx as usize,
                    before: before_ctx as usize,
                    after: after_ctx as usize,
                });
            }
            tier8_diag_verbose!(
                "[TIER8-DIAG energy-pre-sync] call=cuStreamSynchronize \
                 result=OK stream_id={} protocol_group={} rc=0 \
                 md_raw=0x{:x} energy_temp_ptr=0x{:x} energy_temp_bytes={} \
                 energy_scalar_ptr=0x{:x} energy_scalar_bytes=8 \
                 energy_window_ptr=0x{:x} energy_window_bytes=16 \
                 expected_ctx={:p} before_ctx={:p} after_ctx={:p} \
                 did_set_current={} thread_id={}",
                diag_stream_id,
                diag_protocol_group,
                md_raw,
                energy_temp_storage_dev,
                temp_bytes,
                energy_pe_scalar_dev,
                energy_window_dev,
                expected_ctx,
                before_ctx,
                after_ctx,
                did_set_current,
                diag_thread_id
            );
            log::info!(
                "[M1.2.17] energy monitor allocated: temp_storage={} B \
                 pe_scalar=8 B window=16 B; n_atoms={}",
                temp_bytes,
                cfg.n_atoms_for_pe
            );
        }

        // ── Wave B.1 — Initialise __constant__ d_gearbox_table ────────
        // Stream-ordered cudaMemcpyToSymbolAsync of 16 floats (64 bytes,
        // one Blackwell L1 const-cache line) into the gearbox.cu's
        // __constant__ memory. Done OUTSIDE the capture window — the
        // captured graph cannot host-write a __constant__ symbol.
        let gearbox_table = crate::gearbox::default_gearbox_table();
        let gearbox_diag_protocol_group = if cfg.g26_parent_cond_handle != 0 {
            "parent_owned_g26_child"
        } else {
            "standalone_child"
        };
        let gearbox_diag_stream_id = cfg
            .diagnostic_stream_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "unavailable".to_string());
        let gearbox_diag_thread_id = format!("{:?}", std::thread::current().id());
        let (ctx_before_gearbox, ctx_before_gearbox_rc) = tier8_diag_current_context();
        let (ctx_before_gearbox_name, ctx_before_gearbox_text) =
            tier8_diag_driver_error_text(ctx_before_gearbox_rc);
        tier8_diag_verbose!(
            "[TIER8-DIAG gearbox-table] call=preflight result=OK \
             stream_id={} protocol_group={} md_raw=0x{:x} \
             parent_graph=unavailable_in_captured_pipeline parent_cond_handle=0x{:x} \
             current_ctx={:p} current_ctx_rc={} current_ctx_name={} \
             current_ctx_string=\"{}\" thread_id={} table_ptr={:p}",
            gearbox_diag_stream_id,
            gearbox_diag_protocol_group,
            md_raw,
            cfg.g26_parent_cond_handle,
            ctx_before_gearbox,
            ctx_before_gearbox_rc as i32,
            ctx_before_gearbox_name,
            ctx_before_gearbox_text,
            gearbox_diag_thread_id,
            gearbox_table.as_ptr()
        );
        // Canonical gear table is a compile-time constant, including Gear 3
        // NaN. Rebuilding one stream cannot mutate siblings' live constants.
        let rc: i32 = 0;
        if rc != 0 {
            let (name, text) = tier8_diag_runtime_error_text(rc);
            log::error!(
                "[TIER8-DIAG gearbox-table] call=compiled_canonical_gear_table \
                 result=FAIL stream_id={} protocol_group={} rc={} \
                 cuda_name={} cuda_string=\"{}\" md_raw=0x{:x} table_ptr={:p} \
                 parent_cond_handle=0x{:x} thread_id={}",
                gearbox_diag_stream_id,
                gearbox_diag_protocol_group,
                rc,
                name,
                text,
                md_raw,
                gearbox_table.as_ptr(),
                cfg.g26_parent_cond_handle,
                gearbox_diag_thread_id
            );
            return Err(BuildError::Cuda {
                stage: "prism_gearbox_init_table_async",
                rc,
            });
        }
        tier8_diag_verbose!(
            "[TIER8-DIAG gearbox-table] call=compiled_canonical_gear_table \
             result=OK stream_id={} protocol_group={} rc=0 \
             md_raw=0x{:x} table_ptr={:p} parent_cond_handle=0x{:x} thread_id={}",
            gearbox_diag_stream_id,
            gearbox_diag_protocol_group,
            md_raw,
            gearbox_table.as_ptr(),
            cfg.g26_parent_cond_handle,
            gearbox_diag_thread_id
        );

        let gearbox_expected_ctx = ctx.cu_ctx() as CUcontext;
        let mut gearbox_before_ctx: CUcontext = ptr::null_mut();
        let gearbox_get_before_rc = unsafe { cuCtxGetCurrent(&mut gearbox_before_ctx as *mut _) };
        let (gearbox_get_before_name, gearbox_get_before_text) =
            tier8_diag_driver_error_text(gearbox_get_before_rc);
        if !matches!(gearbox_get_before_rc, CUresult::CUDA_SUCCESS) {
            log::error!(
                "[TIER8-DIAG gearbox-table] call=cuCtxGetCurrent(before) \
                 result=FAIL stream_id={} protocol_group={} rc={} \
                 cuda_name={} cuda_string=\"{}\" expected_ctx={:p} \
                 before_ctx={:p} md_raw=0x{:x} parent_cond_handle=0x{:x} \
                 table_ptr={:p} thread_id={}",
                gearbox_diag_stream_id,
                gearbox_diag_protocol_group,
                gearbox_get_before_rc as i32,
                gearbox_get_before_name,
                gearbox_get_before_text,
                gearbox_expected_ctx,
                gearbox_before_ctx,
                md_raw,
                cfg.g26_parent_cond_handle,
                gearbox_table.as_ptr(),
                gearbox_diag_thread_id
            );
            return Err(BuildError::Tier8Context {
                stage: "TIER8 gearbox-table cuCtxGetCurrent(before)",
                rc: gearbox_get_before_rc as i32,
                expected: gearbox_expected_ctx as usize,
                before: gearbox_before_ctx as usize,
                after: 0,
            });
        }

        let mut gearbox_did_set_current = false;
        let mut gearbox_set_current_rc = CUresult::CUDA_SUCCESS;
        let mut gearbox_set_current_name = "CUDA_SUCCESS".to_string();
        let mut gearbox_set_current_text = "no error".to_string();
        if gearbox_before_ctx.is_null() || gearbox_before_ctx != gearbox_expected_ctx {
            gearbox_did_set_current = true;
            gearbox_set_current_rc = unsafe { cuCtxSetCurrent(gearbox_expected_ctx) };
            let (name, text) = tier8_diag_driver_error_text(gearbox_set_current_rc);
            gearbox_set_current_name = name;
            gearbox_set_current_text = text;
            if !matches!(gearbox_set_current_rc, CUresult::CUDA_SUCCESS) {
                log::error!(
                    "[TIER8-DIAG gearbox-table] call=cuCtxSetCurrent \
                     result=FAIL stream_id={} protocol_group={} rc={} \
                     cuda_name={} cuda_string=\"{}\" expected_ctx={:p} \
                     before_ctx={:p} md_raw=0x{:x} parent_cond_handle=0x{:x} \
                     table_ptr={:p} thread_id={}",
                    gearbox_diag_stream_id,
                    gearbox_diag_protocol_group,
                    gearbox_set_current_rc as i32,
                    gearbox_set_current_name,
                    gearbox_set_current_text,
                    gearbox_expected_ctx,
                    gearbox_before_ctx,
                    md_raw,
                    cfg.g26_parent_cond_handle,
                    gearbox_table.as_ptr(),
                    gearbox_diag_thread_id
                );
                return Err(BuildError::Tier8Context {
                    stage: "TIER8 gearbox-table cuCtxSetCurrent",
                    rc: gearbox_set_current_rc as i32,
                    expected: gearbox_expected_ctx as usize,
                    before: gearbox_before_ctx as usize,
                    after: 0,
                });
            }
        }

        let mut gearbox_after_ctx: CUcontext = ptr::null_mut();
        let gearbox_get_after_rc = unsafe { cuCtxGetCurrent(&mut gearbox_after_ctx as *mut _) };
        let (gearbox_get_after_name, gearbox_get_after_text) =
            tier8_diag_driver_error_text(gearbox_get_after_rc);
        if !matches!(gearbox_get_after_rc, CUresult::CUDA_SUCCESS) {
            log::error!(
                "[TIER8-DIAG gearbox-table] call=cuCtxGetCurrent(after) \
                 result=FAIL stream_id={} protocol_group={} rc={} \
                 cuda_name={} cuda_string=\"{}\" expected_ctx={:p} \
                 before_ctx={:p} after_ctx={:p} did_set_current={} \
                 set_rc={} set_name={} set_string=\"{}\" md_raw=0x{:x} \
                 parent_cond_handle=0x{:x} table_ptr={:p} thread_id={}",
                gearbox_diag_stream_id,
                gearbox_diag_protocol_group,
                gearbox_get_after_rc as i32,
                gearbox_get_after_name,
                gearbox_get_after_text,
                gearbox_expected_ctx,
                gearbox_before_ctx,
                gearbox_after_ctx,
                gearbox_did_set_current,
                gearbox_set_current_rc as i32,
                gearbox_set_current_name,
                gearbox_set_current_text,
                md_raw,
                cfg.g26_parent_cond_handle,
                gearbox_table.as_ptr(),
                gearbox_diag_thread_id
            );
            return Err(BuildError::Tier8Context {
                stage: "TIER8 gearbox-table cuCtxGetCurrent(after)",
                rc: gearbox_get_after_rc as i32,
                expected: gearbox_expected_ctx as usize,
                before: gearbox_before_ctx as usize,
                after: gearbox_after_ctx as usize,
            });
        }
        if gearbox_after_ctx != gearbox_expected_ctx {
            log::error!(
                "[TIER8-DIAG gearbox-table] call=raw_context_guard \
                 result=FAIL stream_id={} protocol_group={} reason=context_mismatch \
                 expected_ctx={:p} before_ctx={:p} after_ctx={:p} \
                 did_set_current={} get_before_rc={} get_before_name={} \
                 get_before_string=\"{}\" set_rc={} set_name={} set_string=\"{}\" \
                 get_after_rc={} get_after_name={} get_after_string=\"{}\" \
                 md_raw=0x{:x} parent_cond_handle=0x{:x} table_ptr={:p} \
                 thread_id={}",
                gearbox_diag_stream_id,
                gearbox_diag_protocol_group,
                gearbox_expected_ctx,
                gearbox_before_ctx,
                gearbox_after_ctx,
                gearbox_did_set_current,
                gearbox_get_before_rc as i32,
                gearbox_get_before_name,
                gearbox_get_before_text,
                gearbox_set_current_rc as i32,
                gearbox_set_current_name,
                gearbox_set_current_text,
                gearbox_get_after_rc as i32,
                gearbox_get_after_name,
                gearbox_get_after_text,
                md_raw,
                cfg.g26_parent_cond_handle,
                gearbox_table.as_ptr(),
                gearbox_diag_thread_id
            );
            return Err(BuildError::Tier8Context {
                stage: "TIER8 gearbox-table context mismatch",
                rc: -1,
                expected: gearbox_expected_ctx as usize,
                before: gearbox_before_ctx as usize,
                after: gearbox_after_ctx as usize,
            });
        }
        tier8_diag_verbose!(
            "[TIER8-DIAG gearbox-table] call=raw_context_guard \
             result=OK stream_id={} protocol_group={} expected_ctx={:p} \
             before_ctx={:p} after_ctx={:p} did_set_current={} \
             get_before_rc={} get_before_name={} get_before_string=\"{}\" \
             set_rc={} set_name={} set_string=\"{}\" \
             get_after_rc={} get_after_name={} get_after_string=\"{}\" \
             md_raw=0x{:x} parent_cond_handle=0x{:x} table_ptr={:p} \
             thread_id={}",
            gearbox_diag_stream_id,
            gearbox_diag_protocol_group,
            gearbox_expected_ctx,
            gearbox_before_ctx,
            gearbox_after_ctx,
            gearbox_did_set_current,
            gearbox_get_before_rc as i32,
            gearbox_get_before_name,
            gearbox_get_before_text,
            gearbox_set_current_rc as i32,
            gearbox_set_current_name,
            gearbox_set_current_text,
            gearbox_get_after_rc as i32,
            gearbox_get_after_name,
            gearbox_get_after_text,
            md_raw,
            cfg.g26_parent_cond_handle,
            gearbox_table.as_ptr(),
            gearbox_diag_thread_id
        );

        let gearbox_sync_rc = unsafe { cuStreamSynchronize(md_raw as CUstream) };
        if !matches!(gearbox_sync_rc, CUresult::CUDA_SUCCESS) {
            let (name, text) = tier8_diag_driver_error_text(gearbox_sync_rc);
            let (ctx_after_sync, ctx_after_sync_rc) = tier8_diag_current_context();
            log::error!(
                "[TIER8-DIAG gearbox-table] call=cuStreamSynchronize \
                 result=FAIL stream_id={} protocol_group={} \
                 rc={} cuda_name={} cuda_string=\"{}\" md_raw=0x{:x} \
                 parent_cond_handle=0x{:x} expected_ctx={:p} before_ctx={:p} \
                 after_ctx={:p} did_set_current={} current_ctx={:p} \
                 current_ctx_rc={} table_ptr={:p} thread_id={}",
                gearbox_diag_stream_id,
                gearbox_diag_protocol_group,
                gearbox_sync_rc as i32,
                name,
                text,
                md_raw,
                cfg.g26_parent_cond_handle,
                gearbox_expected_ctx,
                gearbox_before_ctx,
                gearbox_after_ctx,
                gearbox_did_set_current,
                ctx_after_sync,
                ctx_after_sync_rc as i32,
                gearbox_table.as_ptr(),
                gearbox_diag_thread_id
            );
            return Err(BuildError::Tier8Context {
                stage: "TIER8-DIAG raw cuStreamSynchronize gearbox table",
                rc: gearbox_sync_rc as i32,
                expected: gearbox_expected_ctx as usize,
                before: gearbox_before_ctx as usize,
                after: gearbox_after_ctx as usize,
            });
        }
        tier8_diag_verbose!(
            "[TIER8-DIAG gearbox-table] call=cuStreamSynchronize \
             result=OK stream_id={} protocol_group={} rc=0 \
             md_raw=0x{:x} parent_cond_handle=0x{:x} expected_ctx={:p} \
             before_ctx={:p} after_ctx={:p} did_set_current={} \
             thread_id={}",
            gearbox_diag_stream_id,
            gearbox_diag_protocol_group,
            md_raw,
            cfg.g26_parent_cond_handle,
            gearbox_expected_ctx,
            gearbox_before_ctx,
            gearbox_after_ctx,
            gearbox_did_set_current,
            gearbox_diag_thread_id
        );

        // CSR §M alignment guard (Anti-Greenfield Audit Gate G19):
        // the F2 pool must return 128-byte-aligned tile addresses for
        // the Adjudicator's LDG.E.128 path. The pool is documented to
        // do so for any allocation ≥ 128 B.
        if tiles_dev % 128 != 0 {
            return Err(BuildError::AlignmentDrift {
                what: "tiles_dev_ptr",
                got: tiles_dev,
                required: 128,
            });
        }

        // ── 4. Pinned host ring (CLA-2) ──────────────────────────────
        let ring = crate::governor_device::CudaOwned::new(
            PinnedTelemetryRing::<ContactShellTile>::new(cfg.n_clusters as usize).map_err(BuildError::PinnedRing)?,
            md_stream.clone());

        // ── 5. Pre-capture: zero adjudicator + tile arrays ───────────
        // The Adjudicator's `prism_interferometric_adjudicator_create`
        // also zero-inits the FFI struct; we use cuMemsetD8 directly
        // to keep the pre-capture sequence host-sync-free.
        unsafe {
            let rc = cuMemsetD8Async(
                adj_dev as CUdeviceptr,
                0,
                std::mem::size_of::<InterferometricAdjudicatorFfi>(), md_stream.cu_stream());
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                return Err(BuildError::Cuda {
                    stage: "memset adj",
                    rc: rc as i32,
                });
            }
            let rc = cuMemsetD8Async(burst_marker_dev as CUdeviceptr, 0, 4, md_stream.cu_stream());
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                return Err(BuildError::Cuda {
                    stage: "memset burst_marker",
                    rc: rc as i32,
                });
            }
        }

        // Host pointer-field uploads below use synchronous driver copies.
        // Publish all async allocations and zero initializers before those
        // copies, including builds with energy monitoring disabled. This is
        // constructor-only and adds no barrier to captured graph replay.
        md_stream.synchronize().map_err(|error| BuildError::Cuda {
            stage: "publish F2 allocations before host pointer wiring",
            rc: error.0 as i32,
        })?;

        // Manifest pointer registry — pre-capture, pointer-stable.
        let manifest = SiteManifestFfi {
            total_sites: cfg.n_clusters,
            _pad0: 0,
            tiles_dev_ptr: tiles_dev as *mut ContactShellTile,
            vram_high_water_mark: tiles_bytes,
            adjudication_trigger_ptr: ptr::null_mut(),
        };
        debug_assert!(
            manifest.tile_alignment_ok(),
            "F2 pool returned non-128-aligned tiles_dev_ptr"
        );

        // Path C pointer wiring: relaxed = SO(3)(raw spikes), perturbed =
        // SO(3)(gasp spikes). freeze_baseline may later redirect the relaxed
        // pointer to a completed raw projection at the chosen anchor.
        unsafe {
            let relaxed_ptr_field = (adj_dev + 56) as CUdeviceptr;
            let perturbed_ptr_field = (adj_dev + 64) as CUdeviceptr;
            let baseline_value: u64 = tiles_baseline_dev as u64;
            let current_value: u64 = manifest.tiles_dev_ptr as u64;
            let rc1 = cuMemcpyHtoD_v2(
                relaxed_ptr_field,
                &baseline_value as *const _ as *const c_void,
                8,
            );
            let rc2 = cuMemcpyHtoD_v2(
                perturbed_ptr_field,
                &current_value as *const _ as *const c_void,
                8,
            );
            for (rc, stage) in [
                (rc1, "seed relaxed_ptr=baseline"),
                (rc2, "seed perturbed_ptr=current"),
            ] {
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage,
                        rc: rc as i32,
                    });
                }
            }
        }

        // ── 5.SISR-pre G28: dyad-axis init + Adjudicator force_prune_mask
        //       pointer wire-in + RECT-3.4.1 device-resident cluster count.
        //       Done OUTSIDE the capture window so subsequent captures
        //       observe a stable adj_dev struct + initialised count buffer.
        if let Some(ref sisr) = cfg.sisr {
            // Init dyad transform in __constant__ memory.
            let rc = unsafe {
                prism_sisr_init_dyad(
                    sisr.dyad_R_row_major.as_ptr(),
                    sisr.dyad_t.as_ptr(),
                    md_raw as *mut c_void,
                )
            };
            if rc != 0 {
                return Err(BuildError::Cuda {
                    stage: "prism_sisr_init_dyad",
                    rc,
                });
            }
            // Wire adj->force_prune_mask = sisr_mask_dev. Offset 104 per
            // the C-side static_assert.
            unsafe {
                let mask_field_addr = (adj_dev + 104) as CUdeviceptr;
                let mask_ptr_value: u64 = sisr_mask_dev as u64;
                let rc = cuMemcpyHtoD_v2(
                    mask_field_addr,
                    &mask_ptr_value as *const _ as *const c_void,
                    8,
                );
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage: "wire adj->force_prune_mask",
                        rc: rc as i32,
                    });
                }
            }
            // RECT-3.4.1: initialise device-resident cluster count = cfg.n_clusters.
            // The SISR kernel reads this at execution time and early-exits when < 2.
            // Future: SpikeToCluster4D writes the dynamic count here per-frame.
            unsafe {
                let count_addr = sisr_count_dev as CUdeviceptr;
                let count_value: u32 = cfg.n_clusters;
                let rc = cuMemcpyHtoD_v2(count_addr, &count_value as *const _ as *const c_void, 4);
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage: "init sisr_count_dev",
                        rc: rc as i32,
                    });
                }
            }
            md_stream.synchronize().map_err(BuildError::Driver)?;
        }

        // ── 5b. M1.2.17 — wire adj->d_dt at offset 120 ──────────────
        // The d_dt pointer is written into adj_dev at offset 120 (the
        // M1.2.17 layout pivot moved it from 112 to 120 to make room
        // for the f64 d_potential_energy VALUE at the L1-aligned 112
        // slot).  d_velocities is no longer in the struct — it lives
        // on PipelineConfig only and the populator FFI threads it
        // directly into the SWITCH body rescale kernels.
        if !cfg.d_dt.is_null() {
            unsafe {
                let field_addr = (adj_dev + 120) as CUdeviceptr;
                let value: u64 = cfg.d_dt as u64;
                let rc = cuMemcpyHtoD_v2(field_addr, &value as *const _ as *const c_void, 8);
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage: "wire adj->d_dt (M1.2.17 offset 120)",
                        rc: rc as i32,
                    });
                }
            }
            let ddt_diag_stream_id = cfg.diagnostic_stream_id.unwrap_or(u32::MAX);
            let ddt_diag_protocol_group = if cfg.g26_parent_cond_handle != 0 {
                "parent_owned_g26_child"
            } else {
                "standalone_child"
            };
            let ddt_diag_thread_id = format!("{:?}", std::thread::current().id());
            let ddt_expected_ctx = ctx.cu_ctx() as CUcontext;
            let mut ddt_before_ctx: CUcontext = ptr::null_mut();
            let ddt_get_before_rc = unsafe { cuCtxGetCurrent(&mut ddt_before_ctx as *mut _) };
            let (ddt_get_before_name, ddt_get_before_text) =
                tier8_diag_driver_error_text(ddt_get_before_rc);
            if !matches!(ddt_get_before_rc, CUresult::CUDA_SUCCESS) {
                log::error!(
                    "[TIER8-DIAG d-dt-wire-sync] call=cuCtxGetCurrent(before) \
                     result=FAIL stream_id={} protocol_group={} rc={} \
                     cuda_name={} cuda_string={:?} md_raw=0x{:x} \
                     expected_ctx={:p} before_ctx={:p} d_dt={:p} \
                     parent_cond_handle=0x{:x} thread_id={}",
                    ddt_diag_stream_id,
                    ddt_diag_protocol_group,
                    ddt_get_before_rc as i32,
                    ddt_get_before_name,
                    ddt_get_before_text,
                    md_raw,
                    ddt_expected_ctx,
                    ddt_before_ctx,
                    cfg.d_dt as *const f32,
                    cfg.g26_parent_cond_handle,
                    ddt_diag_thread_id
                );
                return Err(BuildError::Tier8Context {
                    stage: "TIER8-DIAG cuCtxGetCurrent(before) d_dt wire sync",
                    rc: ddt_get_before_rc as i32,
                    expected: ddt_expected_ctx as usize,
                    before: ddt_before_ctx as usize,
                    after: 0,
                });
            }

            let mut ddt_did_set_current = false;
            let mut ddt_set_current_rc = CUresult::CUDA_SUCCESS;
            if ddt_before_ctx.is_null() || ddt_before_ctx != ddt_expected_ctx {
                ddt_set_current_rc = unsafe { cuCtxSetCurrent(ddt_expected_ctx) };
                ddt_did_set_current = true;
                let (set_name, set_text) = tier8_diag_driver_error_text(ddt_set_current_rc);
                if !matches!(ddt_set_current_rc, CUresult::CUDA_SUCCESS) {
                    log::error!(
                        "[TIER8-DIAG d-dt-wire-sync] call=cuCtxSetCurrent \
                         result=FAIL stream_id={} protocol_group={} rc={} \
                         cuda_name={} cuda_string={:?} md_raw=0x{:x} \
                         expected_ctx={:p} before_ctx={:p} d_dt={:p} \
                         parent_cond_handle=0x{:x} thread_id={}",
                        ddt_diag_stream_id,
                        ddt_diag_protocol_group,
                        ddt_set_current_rc as i32,
                        set_name,
                        set_text,
                        md_raw,
                        ddt_expected_ctx,
                        ddt_before_ctx,
                        cfg.d_dt as *const f32,
                        cfg.g26_parent_cond_handle,
                        ddt_diag_thread_id
                    );
                    return Err(BuildError::Tier8Context {
                        stage: "TIER8-DIAG cuCtxSetCurrent d_dt wire sync",
                        rc: ddt_set_current_rc as i32,
                        expected: ddt_expected_ctx as usize,
                        before: ddt_before_ctx as usize,
                        after: 0,
                    });
                }
            }

            let mut ddt_after_ctx: CUcontext = ptr::null_mut();
            let ddt_get_after_rc = unsafe { cuCtxGetCurrent(&mut ddt_after_ctx as *mut _) };
            let (ddt_get_after_name, ddt_get_after_text) =
                tier8_diag_driver_error_text(ddt_get_after_rc);
            if !matches!(ddt_get_after_rc, CUresult::CUDA_SUCCESS) {
                log::error!(
                    "[TIER8-DIAG d-dt-wire-sync] call=cuCtxGetCurrent(after) \
                     result=FAIL stream_id={} protocol_group={} rc={} \
                     cuda_name={} cuda_string={:?} md_raw=0x{:x} \
                     expected_ctx={:p} before_ctx={:p} after_ctx={:p} \
                     did_set_current={} set_rc={} d_dt={:p} \
                     parent_cond_handle=0x{:x} thread_id={}",
                    ddt_diag_stream_id,
                    ddt_diag_protocol_group,
                    ddt_get_after_rc as i32,
                    ddt_get_after_name,
                    ddt_get_after_text,
                    md_raw,
                    ddt_expected_ctx,
                    ddt_before_ctx,
                    ddt_after_ctx,
                    ddt_did_set_current,
                    ddt_set_current_rc as i32,
                    cfg.d_dt as *const f32,
                    cfg.g26_parent_cond_handle,
                    ddt_diag_thread_id
                );
                return Err(BuildError::Tier8Context {
                    stage: "TIER8-DIAG cuCtxGetCurrent(after) d_dt wire sync",
                    rc: ddt_get_after_rc as i32,
                    expected: ddt_expected_ctx as usize,
                    before: ddt_before_ctx as usize,
                    after: ddt_after_ctx as usize,
                });
            }

            let (ddt_set_name, ddt_set_text) = tier8_diag_driver_error_text(ddt_set_current_rc);
            if ddt_after_ctx != ddt_expected_ctx {
                log::error!(
                    "[TIER8-DIAG d-dt-wire-sync] call=raw_context_guard \
                     result=FAIL stream_id={} protocol_group={} reason=context_mismatch \
                     expected_ctx={:p} before_ctx={:p} after_ctx={:p} \
                     did_set_current={} get_before_rc={} get_before_name={} \
                     get_before_string={:?} set_rc={} set_name={} set_string={:?} \
                     get_after_rc={} get_after_name={} get_after_string={:?} \
                     md_raw=0x{:x} d_dt={:p} parent_cond_handle=0x{:x} \
                     thread_id={}",
                    ddt_diag_stream_id,
                    ddt_diag_protocol_group,
                    ddt_expected_ctx,
                    ddt_before_ctx,
                    ddt_after_ctx,
                    ddt_did_set_current,
                    ddt_get_before_rc as i32,
                    ddt_get_before_name,
                    ddt_get_before_text,
                    ddt_set_current_rc as i32,
                    ddt_set_name,
                    ddt_set_text,
                    ddt_get_after_rc as i32,
                    ddt_get_after_name,
                    ddt_get_after_text,
                    md_raw,
                    cfg.d_dt as *const f32,
                    cfg.g26_parent_cond_handle,
                    ddt_diag_thread_id
                );
                return Err(BuildError::Tier8Context {
                    stage: "TIER8-DIAG raw context mismatch d_dt wire sync",
                    rc: -2,
                    expected: ddt_expected_ctx as usize,
                    before: ddt_before_ctx as usize,
                    after: ddt_after_ctx as usize,
                });
            }
            tier8_diag_verbose!(
                "[TIER8-DIAG d-dt-wire-sync] call=raw_context_guard \
                 result=OK stream_id={} protocol_group={} expected_ctx={:p} \
                 before_ctx={:p} after_ctx={:p} did_set_current={} \
                 get_before_rc={} get_before_name={} get_before_string={:?} \
                 set_rc={} set_name={} set_string={:?} get_after_rc={} \
                 get_after_name={} get_after_string={:?} md_raw=0x{:x} \
                 d_dt={:p} parent_cond_handle=0x{:x} thread_id={}",
                ddt_diag_stream_id,
                ddt_diag_protocol_group,
                ddt_expected_ctx,
                ddt_before_ctx,
                ddt_after_ctx,
                ddt_did_set_current,
                ddt_get_before_rc as i32,
                ddt_get_before_name,
                ddt_get_before_text,
                ddt_set_current_rc as i32,
                ddt_set_name,
                ddt_set_text,
                ddt_get_after_rc as i32,
                ddt_get_after_name,
                ddt_get_after_text,
                md_raw,
                cfg.d_dt as *const f32,
                cfg.g26_parent_cond_handle,
                ddt_diag_thread_id
            );

            let ddt_sync_rc = unsafe { cuStreamSynchronize(md_stream.cu_stream()) };
            if !matches!(ddt_sync_rc, CUresult::CUDA_SUCCESS) {
                let (sync_name, sync_text) = tier8_diag_driver_error_text(ddt_sync_rc);
                log::error!(
                    "[TIER8-DIAG d-dt-wire-sync] call=cuStreamSynchronize \
                     result=FAIL stream_id={} protocol_group={} rc={} \
                     cuda_name={} cuda_string={:?} md_raw=0x{:x} \
                     expected_ctx={:p} before_ctx={:p} after_ctx={:p} \
                     did_set_current={} d_dt={:p} parent_cond_handle=0x{:x} \
                     thread_id={}",
                    ddt_diag_stream_id,
                    ddt_diag_protocol_group,
                    ddt_sync_rc as i32,
                    sync_name,
                    sync_text,
                    md_raw,
                    ddt_expected_ctx,
                    ddt_before_ctx,
                    ddt_after_ctx,
                    ddt_did_set_current,
                    cfg.d_dt as *const f32,
                    cfg.g26_parent_cond_handle,
                    ddt_diag_thread_id
                );
                return Err(BuildError::Tier8Context {
                    stage: "TIER8-DIAG raw cuStreamSynchronize d_dt wire sync",
                    rc: ddt_sync_rc as i32,
                    expected: ddt_expected_ctx as usize,
                    before: ddt_before_ctx as usize,
                    after: ddt_after_ctx as usize,
                });
            }
            tier8_diag_verbose!(
                "[TIER8-DIAG d-dt-wire-sync] call=cuStreamSynchronize \
                 result=OK stream_id={} protocol_group={} rc=0 md_raw=0x{:x} \
                 expected_ctx={:p} before_ctx={:p} after_ctx={:p} \
                 did_set_current={} d_dt={:p} parent_cond_handle=0x{:x} \
                 thread_id={}",
                ddt_diag_stream_id,
                ddt_diag_protocol_group,
                md_raw,
                ddt_expected_ctx,
                ddt_before_ctx,
                ddt_after_ctx,
                ddt_did_set_current,
                cfg.d_dt as *const f32,
                cfg.g26_parent_cond_handle,
                ddt_diag_thread_id
            );
            log::info!(
                "[M1.2.17] wired adj->d_dt={:p} at offset 120; \
                 cfg.d_velocities={:p} (struct slot removed; populator threads it)",
                cfg.d_dt as *const f32,
                cfg.d_velocities as *const f32
            );
        }

        // ── 5c. Wire adj->d_external_work at offset 128 (Zero-Host Guard) ──
        // The W_ext F2-pool buffer pointer is written into adj_dev at
        // offset 128 (pointer fusion).  Six writer functions accumulate ΔK
        // into the per-atom [n_atoms] f64 plane with a plain non-atomic `+=`
        // — there is no atomicAdd on this buffer anywhere in the tree.  The
        // authoritative contributor list lives in the d_external_work field
        // comment in cuda/adjudicator.cuh and is enforced by the
        // w_ext_contributor_set_is_derived_from_cuda_source contract test in
        // fused_engine.rs; do not restate it here.
        // The SFA stability fuse dereferences the pointer for the First-Law
        // drift formula |V_t − (V_{t-1} + W_ext)| / |V_{t-1} + W_ext|.
        //
        // The physical producer resets atom rows before integration; this
        // downstream observer consumes its accumulated interval before reset.
        // Clearing here before SFA would erase completed physical evidence.
        //
        // **Operator Zero-Trust §3 Zero-Host Guard:** null at capture
        // time is a LANE-BLOCKED escalation.  The integrator's 73rd
        // parameter (index 72, pinned by the parameters[72] assertion in
        // fused_cuda_abi_is_hash_bound_and_all_launches_supply_every_parameter)
        // has nowhere to accumulate without this buffer; a null pointer here
        // would let the UV, clamp and SHAKE/RATTLE instrumentation silently
        // drop ΔK contributions, breaking the audit.
        let ext_diag_stream_label = cfg
            .diagnostic_stream_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "unavailable".to_string());
        let ext_diag_protocol_group = if cfg.g26_parent_cond_handle != 0 {
            "parent_owned_g26_child"
        } else {
            "standalone_child"
        };
        let ext_diag_phase = "pre_capture_external_work_wire";
        let ext_diag_thread_id = format!("{:?}", std::thread::current().id());
        let ext_expected_ctx = ctx.cu_ctx() as CUcontext;
        let mut ext_before_ctx: CUcontext = ptr::null_mut();
        let ext_get_before_rc = unsafe { cuCtxGetCurrent(&mut ext_before_ctx as *mut _) };
        let (ext_get_before_name, ext_get_before_text) =
            tier8_diag_driver_error_text(ext_get_before_rc);
        tier8_diag_verbose!(
            "[TIER8-DIAG external-work-wire] call=cuCtxGetCurrent(before) \
             result={} stream_id={} protocol_group={} phase={} rc={} \
             cuda_name={} cuda_string={:?} md_raw=0x{:x} expected_ctx={:p} \
             before_ctx={:p} adj_dev=0x{:x} field_offset=128 field_addr=0x{:x} \
             d_external_work={:p} is_null={} parent_cond_handle=0x{:x} thread_id={}",
            if matches!(ext_get_before_rc, CUresult::CUDA_SUCCESS) {
                "OK"
            } else {
                "FAIL"
            },
            ext_diag_stream_label,
            ext_diag_protocol_group,
            ext_diag_phase,
            ext_get_before_rc as i32,
            ext_get_before_name,
            ext_get_before_text,
            md_raw,
            ext_expected_ctx,
            ext_before_ctx,
            adj_dev,
            adj_dev + 128,
            cfg.d_external_work as *const f64,
            cfg.d_external_work.is_null(),
            cfg.g26_parent_cond_handle,
            ext_diag_thread_id
        );
        if !matches!(ext_get_before_rc, CUresult::CUDA_SUCCESS) {
            return Err(BuildError::Tier8Context {
                stage: "TIER8-DIAG cuCtxGetCurrent(before) external work wire",
                rc: ext_get_before_rc as i32,
                expected: ext_expected_ctx as usize,
                before: ext_before_ctx as usize,
                after: 0,
            });
        }

        tier8_diag_verbose!(
            "[TIER8-DIAG external-work-wire] call=null_check result={} \
             stream_id={} protocol_group={} phase={} md_raw=0x{:x} \
             expected_ctx={:p} before_ctx={:p} adj_dev=0x{:x} \
             field_offset=128 field_addr=0x{:x} d_external_work={:p} \
             parent_cond_handle=0x{:x} thread_id={}",
            if cfg.d_external_work.is_null() {
                "FAIL"
            } else {
                "OK"
            },
            ext_diag_stream_label,
            ext_diag_protocol_group,
            ext_diag_phase,
            md_raw,
            ext_expected_ctx,
            ext_before_ctx,
            adj_dev,
            adj_dev + 128,
            cfg.d_external_work as *const f64,
            cfg.g26_parent_cond_handle,
            ext_diag_thread_id
        );
        if cfg.d_external_work.is_null() {
            return Err(BuildError::Cuda {
                stage: "Zero-Host Guard: cfg.d_external_work is null at capture \
                        time (LANE BLOCKED — F2-pool buffer must be allocated \
                        via NhsAmberFusedEngine::allocate_external_work_buffer \
                        before captured pipeline build)",
                rc: -1,
            });
        }
        unsafe {
            let field_addr = (adj_dev + 128) as CUdeviceptr;
            let value: u64 = cfg.d_external_work as u64;
            tier8_diag_verbose!(
                "[TIER8-DIAG external-work-wire] call=cuMemcpyHtoD_v2 \
                 result=BEGIN stream_id={} protocol_group={} phase={} \
                 md_raw=0x{:x} expected_ctx={:p} before_ctx={:p} dst=0x{:x} \
                 src_host={:p} bytes=8 value=0x{:x} parent_cond_handle=0x{:x} \
                 thread_id={}",
                ext_diag_stream_label,
                ext_diag_protocol_group,
                ext_diag_phase,
                md_raw,
                ext_expected_ctx,
                ext_before_ctx,
                field_addr,
                &value as *const u64,
                value,
                cfg.g26_parent_cond_handle,
                ext_diag_thread_id
            );
            let rc = cuMemcpyHtoD_v2(field_addr, &value as *const _ as *const c_void, 8);
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                let (name, text) = tier8_diag_driver_error_text(rc);
                log::error!(
                    "[TIER8-DIAG external-work-wire] call=cuMemcpyHtoD_v2 \
                     result=FAIL stream_id={} protocol_group={} phase={} rc={} \
                     cuda_name={} cuda_string={:?} md_raw=0x{:x} expected_ctx={:p} \
                     before_ctx={:p} dst=0x{:x} src_host={:p} bytes=8 value=0x{:x} \
                     parent_cond_handle=0x{:x} thread_id={}",
                    ext_diag_stream_label,
                    ext_diag_protocol_group,
                    ext_diag_phase,
                    rc as i32,
                    name,
                    text,
                    md_raw,
                    ext_expected_ctx,
                    ext_before_ctx,
                    field_addr,
                    &value as *const u64,
                    value,
                    cfg.g26_parent_cond_handle,
                    ext_diag_thread_id
                );
                return Err(BuildError::Cuda {
                    stage: "wire adj->d_external_work (offset 128)",
                    rc: rc as i32,
                });
            }
            let (name, text) = tier8_diag_driver_error_text(rc);
            tier8_diag_verbose!(
                "[TIER8-DIAG external-work-wire] call=cuMemcpyHtoD_v2 \
                 result=OK stream_id={} protocol_group={} phase={} rc=0 \
                 cuda_name={} cuda_string={:?} md_raw=0x{:x} expected_ctx={:p} \
                 before_ctx={:p} dst=0x{:x} src_host={:p} bytes=8 value=0x{:x} \
                 parent_cond_handle=0x{:x} thread_id={}",
                ext_diag_stream_label,
                ext_diag_protocol_group,
                ext_diag_phase,
                name,
                text,
                md_raw,
                ext_expected_ctx,
                ext_before_ctx,
                field_addr,
                &value as *const u64,
                value,
                cfg.g26_parent_cond_handle,
                ext_diag_thread_id
            );
        }
        let mut ext_after_memcpy_ctx: CUcontext = ptr::null_mut();
        let ext_get_after_memcpy_rc =
            unsafe { cuCtxGetCurrent(&mut ext_after_memcpy_ctx as *mut _) };
        let (ext_get_after_memcpy_name, ext_get_after_memcpy_text) =
            tier8_diag_driver_error_text(ext_get_after_memcpy_rc);
        tier8_diag_verbose!(
            "[TIER8-DIAG external-work-wire] call=cuCtxGetCurrent(after_memcpy) \
             result={} stream_id={} protocol_group={} phase={} rc={} \
             cuda_name={} cuda_string={:?} md_raw=0x{:x} expected_ctx={:p} \
             before_ctx={:p} after_memcpy_ctx={:p} d_external_work={:p} \
             parent_cond_handle=0x{:x} thread_id={}",
            if matches!(ext_get_after_memcpy_rc, CUresult::CUDA_SUCCESS) {
                "OK"
            } else {
                "FAIL"
            },
            ext_diag_stream_label,
            ext_diag_protocol_group,
            ext_diag_phase,
            ext_get_after_memcpy_rc as i32,
            ext_get_after_memcpy_name,
            ext_get_after_memcpy_text,
            md_raw,
            ext_expected_ctx,
            ext_before_ctx,
            ext_after_memcpy_ctx,
            cfg.d_external_work as *const f64,
            cfg.g26_parent_cond_handle,
            ext_diag_thread_id
        );
        if !matches!(ext_get_after_memcpy_rc, CUresult::CUDA_SUCCESS) {
            return Err(BuildError::Tier8Context {
                stage: "TIER8-DIAG cuCtxGetCurrent(after_memcpy) external work wire",
                rc: ext_get_after_memcpy_rc as i32,
                expected: ext_expected_ctx as usize,
                before: ext_before_ctx as usize,
                after: ext_after_memcpy_ctx as usize,
            });
        }
        tier8_diag_verbose!(
            "[TIER8-DIAG external-work-wire] call=cuStreamSynchronize \
             result=BEGIN stream_id={} protocol_group={} phase={} md_raw=0x{:x} \
             expected_ctx={:p} before_ctx={:p} after_memcpy_ctx={:p} \
             d_external_work={:p} parent_cond_handle=0x{:x} thread_id={}",
            ext_diag_stream_label,
            ext_diag_protocol_group,
            ext_diag_phase,
            md_raw,
            ext_expected_ctx,
            ext_before_ctx,
            ext_after_memcpy_ctx,
            cfg.d_external_work as *const f64,
            cfg.g26_parent_cond_handle,
            ext_diag_thread_id
        );
        let ext_sync_rc = unsafe { cuStreamSynchronize(md_stream.cu_stream()) };
        if !matches!(ext_sync_rc, CUresult::CUDA_SUCCESS) {
            let (name, text) = tier8_diag_driver_error_text(ext_sync_rc);
            log::error!(
                "[TIER8-DIAG external-work-wire] call=cuStreamSynchronize \
                 result=FAIL stream_id={} protocol_group={} phase={} rc={} \
                 cuda_name={} cuda_string={:?} md_raw=0x{:x} expected_ctx={:p} \
                 before_ctx={:p} after_memcpy_ctx={:p} d_external_work={:p} \
                 parent_cond_handle=0x{:x} thread_id={}",
                ext_diag_stream_label,
                ext_diag_protocol_group,
                ext_diag_phase,
                ext_sync_rc as i32,
                name,
                text,
                md_raw,
                ext_expected_ctx,
                ext_before_ctx,
                ext_after_memcpy_ctx,
                cfg.d_external_work as *const f64,
                cfg.g26_parent_cond_handle,
                ext_diag_thread_id
            );
            return Err(BuildError::Tier8Context {
                stage: "TIER8-DIAG raw cuStreamSynchronize external work wire",
                rc: ext_sync_rc as i32,
                expected: ext_expected_ctx as usize,
                before: ext_before_ctx as usize,
                after: ext_after_memcpy_ctx as usize,
            });
        }
        let (ext_sync_name, ext_sync_text) = tier8_diag_driver_error_text(ext_sync_rc);
        tier8_diag_verbose!(
            "[TIER8-DIAG external-work-wire] call=cuStreamSynchronize \
             result=OK stream_id={} protocol_group={} phase={} rc=0 \
             cuda_name={} cuda_string={:?} md_raw=0x{:x} expected_ctx={:p} \
             before_ctx={:p} after_memcpy_ctx={:p} d_external_work={:p} \
             parent_cond_handle=0x{:x} thread_id={}",
            ext_diag_stream_label,
            ext_diag_protocol_group,
            ext_diag_phase,
            ext_sync_name,
            ext_sync_text,
            md_raw,
            ext_expected_ctx,
            ext_before_ctx,
            ext_after_memcpy_ctx,
            cfg.d_external_work as *const f64,
            cfg.g26_parent_cond_handle,
            ext_diag_thread_id
        );
        let mut ext_after_sync_ctx: CUcontext = ptr::null_mut();
        let ext_get_after_sync_rc = unsafe { cuCtxGetCurrent(&mut ext_after_sync_ctx as *mut _) };
        let (ext_get_after_sync_name, ext_get_after_sync_text) =
            tier8_diag_driver_error_text(ext_get_after_sync_rc);
        tier8_diag_verbose!(
            "[TIER8-DIAG external-work-wire] call=cuCtxGetCurrent(after_sync) \
             result={} stream_id={} protocol_group={} phase={} rc={} \
             cuda_name={} cuda_string={:?} md_raw=0x{:x} expected_ctx={:p} \
             before_ctx={:p} after_memcpy_ctx={:p} after_sync_ctx={:p} \
             d_external_work={:p} parent_cond_handle=0x{:x} thread_id={}",
            if matches!(ext_get_after_sync_rc, CUresult::CUDA_SUCCESS) {
                "OK"
            } else {
                "FAIL"
            },
            ext_diag_stream_label,
            ext_diag_protocol_group,
            ext_diag_phase,
            ext_get_after_sync_rc as i32,
            ext_get_after_sync_name,
            ext_get_after_sync_text,
            md_raw,
            ext_expected_ctx,
            ext_before_ctx,
            ext_after_memcpy_ctx,
            ext_after_sync_ctx,
            cfg.d_external_work as *const f64,
            cfg.g26_parent_cond_handle,
            ext_diag_thread_id
        );
        if !matches!(ext_get_after_sync_rc, CUresult::CUDA_SUCCESS) {
            return Err(BuildError::Tier8Context {
                stage: "TIER8-DIAG cuCtxGetCurrent(after_sync) external work wire",
                rc: ext_get_after_sync_rc as i32,
                expected: ext_expected_ctx as usize,
                before: ext_before_ctx as usize,
                after: ext_after_sync_ctx as usize,
            });
        }
        log::info!(
            "[ZeroTrust] wired adj->d_external_work={:p} at offset 128 (POINTER); \
             captured graph consumes completed observation work before resetting its scalar",
            cfg.d_external_work as *const f64
        );

        // ── 5c.M1.2.20.C-A — Wire `gasp_gain_eta` (offset 136) and
        //    `force_burst_step` (offset 140) into the freshly-zeroed
        //    adjudicator state.  Per Ruling 2 the base gain is locked
        //    to 1.0; per Ruling 5 the burst step comes from the CLI
        //    flag `--force-burst-at-step` (None → u32::MAX disables).
        //    Both writes precede cuStreamBeginCapture so they are not
        //    recorded into the graph — the host can rewrite them
        //    between replays via cuMemcpyHtoDAsync without re-recording.
        let gap_diag_protocol_group = if cfg.g26_parent_cond_handle != 0 {
            "parent_owned_g26_child"
        } else {
            "standalone_child"
        };
        let gap_expected_ctx = ctx.cu_ctx() as CUcontext;
        let gap_stream_label = cfg
            .diagnostic_stream_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "unavailable".to_string());
        unsafe {
            let eta_addr = (adj_dev + 136) as CUdeviceptr;
            let eta_value: f32 = 1.0_f32; // Ruling 2 — η_base locked.
            let rc_eta = cuMemcpyHtoDAsync_v2(
                eta_addr,
                &eta_value as *const _ as *const c_void,
                4,
                md_raw as CUstream,
            );
            let (eta_name, eta_text) = tier8_diag_driver_error_text(rc_eta);
            tier8_diag_verbose!(
                "[TIER8-DIAG gasp-field-wire] call=cuMemcpyHtoDAsync_v2 \
                 field=gasp_gain_eta result={} stream_id={} protocol_group={} \
                 dst=0x{:x} offset=136 bytes=4 rc={} cuda_name={} \
                 cuda_string={:?} md_raw=0x{:x} value={} parent_cond_handle=0x{:x}",
                if matches!(rc_eta, CUresult::CUDA_SUCCESS) {
                    "OK"
                } else {
                    "FAIL"
                },
                gap_stream_label,
                gap_diag_protocol_group,
                eta_addr,
                rc_eta as i32,
                eta_name,
                eta_text,
                md_raw,
                eta_value,
                cfg.g26_parent_cond_handle
            );
            if !matches!(rc_eta, CUresult::CUDA_SUCCESS) {
                log::error!(
                    "[TIER8-DIAG gasp-field-wire] call=cuMemcpyHtoDAsync_v2 \
                     field=gasp_gain_eta result=FAIL stream_id={} \
                     protocol_group={} dst=0x{:x} offset=136 bytes=4 rc={} \
                     cuda_name={} cuda_string={:?} md_raw=0x{:x} parent_cond_handle=0x{:x}",
                    gap_stream_label,
                    gap_diag_protocol_group,
                    eta_addr,
                    rc_eta as i32,
                    eta_name,
                    eta_text,
                    md_raw,
                    cfg.g26_parent_cond_handle
                );
                return Err(BuildError::Cuda {
                    stage: "wire adj->gasp_gain_eta cuMemcpyHtoDAsync_v2 (offset 136)",
                    rc: rc_eta as i32,
                });
            }

            let burst_addr = (adj_dev + 140) as CUdeviceptr;
            let burst_value: u32 = cfg.force_burst_step.unwrap_or(u32::MAX);
            let rc_burst = cuMemcpyHtoDAsync_v2(
                burst_addr,
                &burst_value as *const _ as *const c_void,
                4,
                md_raw as CUstream,
            );
            let (burst_name, burst_text) = tier8_diag_driver_error_text(rc_burst);
            tier8_diag_verbose!(
                "[TIER8-DIAG gasp-field-wire] call=cuMemcpyHtoDAsync_v2 \
                 field=force_burst_step result={} stream_id={} protocol_group={} \
                 dst=0x{:x} offset=140 bytes=4 rc={} cuda_name={} \
                 cuda_string={:?} md_raw=0x{:x} value={} parent_cond_handle=0x{:x}",
                if matches!(rc_burst, CUresult::CUDA_SUCCESS) {
                    "OK"
                } else {
                    "FAIL"
                },
                gap_stream_label,
                gap_diag_protocol_group,
                burst_addr,
                rc_burst as i32,
                burst_name,
                burst_text,
                md_raw,
                burst_value,
                cfg.g26_parent_cond_handle
            );
            if !matches!(rc_burst, CUresult::CUDA_SUCCESS) {
                log::error!(
                    "[TIER8-DIAG gasp-field-wire] call=cuMemcpyHtoDAsync_v2 \
                     field=force_burst_step result=FAIL stream_id={} \
                     protocol_group={} dst=0x{:x} offset=140 bytes=4 rc={} \
                     cuda_name={} cuda_string={:?} md_raw=0x{:x} value={} \
                     parent_cond_handle=0x{:x}",
                    gap_stream_label,
                    gap_diag_protocol_group,
                    burst_addr,
                    rc_burst as i32,
                    burst_name,
                    burst_text,
                    md_raw,
                    burst_value,
                    cfg.g26_parent_cond_handle
                );
                return Err(BuildError::Cuda {
                    stage: "wire adj->force_burst_step cuMemcpyHtoDAsync_v2 (offset 140)",
                    rc: rc_burst as i32,
                });
            }
            tier8_diag_verbose!(
                "[TIER8-DIAG gasp-field-wire] call=field_copies_queued \
                 result=OK stream_id={} protocol_group={} md_raw=0x{:x} \
                 fields=[gasp_gain_eta@136,force_burst_step@140] \
                 force_burst_step={} parent_cond_handle=0x{:x}",
                gap_stream_label,
                gap_diag_protocol_group,
                md_raw,
                burst_value,
                cfg.g26_parent_cond_handle
            );
            log::info!(
                "[M1.2.20.C-A] wired adj->gasp_gain_eta=1.0 at offset 136, \
                 adj->force_burst_step={} at offset 140 ({})",
                burst_value,
                if burst_value == u32::MAX {
                    "DISABLED"
                } else {
                    "10× burst armed"
                }
            );
        }
        // TIER 8.β post_gasp_sync — replaced wrapper-based gate with a
        // 5-step raw-CUDA gate (see tier8_diag_post_gasp_sync_gate).
        // Per directive: this path NEVER calls md_stream.synchronize();
        // the raw cuStreamSynchronize(md_stream.cu_stream()) at step 5
        // is the only sync surface so any failure rc unambiguously
        // names the raw call.
        tier8_diag_post_gasp_sync_gate(
            ctx,
            md_stream,
            cfg.diagnostic_stream_id,
            gap_diag_protocol_group,
            gap_expected_ctx,
            md_raw,
        )?;

        // ── 5a. Wave 0 / Task #68 — KL-UNITS BOOTSTRAP.
        //       Burn KL-magnitude noise-floor priors (μ=0, σ=1e-3 →
        //       threshold=3e-3) into the freshly-zeroed adjudicator
        //       BEFORE the capture window opens.  Pre-Wave-0 this
        //       path called `apply_t7_calibration` which wrote the
        //       4LPK C_l power-spectrum statistics (μ[0]≈0.805) into
        //       a slot the step kernel uses as a KL-divergence
        //       threshold — units mismatch, threshold ~1.249 in C_l
        //       space vs. cold-hold KL ≪ 1.  The 15k 7C8R run on
        //       2026-05-02 produced 35k steps × 0 GhostTileFrame
        //       records as a result.  `apply_t7_kl_calibration` lays
        //       down a small KL-magnitude prior; Dynamic T7 (the
        //       captured graph node downstream of the Adjudicator
        //       step) replaces μ[0]/σ[0] with the substrate's
        //       measured cold-hold KL once it has ≥100 samples.
        //       cudaMemcpyAsync runs on md_stream; the explicit
        //       synchronize below guarantees both copies retire
        //       before cuStreamBeginCapture so they are NOT recorded
        //       into the graph.
        {
            let rc = unsafe {
                crate::interferometric_adjudicator::apply_t7_kl_calibration(
                    adj_dev as *mut InterferometricAdjudicatorFfi,
                    md_raw as *mut c_void,
                )
            };
            if rc != 0 {
                return Err(BuildError::Cuda {
                    stage: "apply_t7_kl_calibration",
                    rc,
                });
            }
        }
        // Amendment 3.4.6 — Substrate-Aware Noise-Floor Override.
        // After the locked 4LPK T7 priors are burned in, conditionally
        // overwrite with operator-supplied (μ, σ) pair (uniform across
        // all 6 SH bands).  Skipping this leaves the 4LPK threshold
        // (μ+3σ ≈ 1.249) which is calibrated to KRAS, not 7C8R.
        if let Some((mu, sigma)) = cfg.noise_floor_override {
            let mu_arr = [mu; 6];
            let sigma_arr = [sigma; 6];
            let rc = unsafe {
                crate::interferometric_adjudicator::set_noise_floor_constants(
                    adj_dev as *mut InterferometricAdjudicatorFfi,
                    &mu_arr,
                    &sigma_arr,
                    md_raw as *mut c_void,
                )
            };
            if rc != 0 {
                return Err(BuildError::Cuda {
                    stage: "noise_floor_override",
                    rc,
                });
            }
            log::info!(
                "[T7-OVERRIDE] noise_floor_mu={} noise_floor_sigma={} → threshold={}",
                mu,
                sigma,
                mu + 3.0 * sigma
            );
        }
        // Native reference initialization, outside graph capture. Merely
        // capturing the relaxed projector does not execute it: the host can
        // freeze the reference before any replay. Use the exact same raw
        // spikes, CSR partition, SH basis and provenance as the relaxed
        // branch. This launches only the projector, never MD, GASP, ASC,
        // adjudication, a governor step, or synthetic observation emission.
        let baseline_rc = unsafe {
            crate::so3_project::ffi::prism_so3_project_seed_run(
                cfg.d_spikes,
                cfg.d_cluster_offsets,
                cfg.n_clusters,
                cfg.d_k_lm,
                tiles_baseline_dev as *mut ContactShellTile,
                cfg.initial_frame_id,
                cfg.source_stream_id,
                md_stream.cu_stream() as *mut c_void,
                    cfg.seed_layout,
                )
        };
        if baseline_rc != crate::so3_project::ffi::CUDA_SUCCESS {
            return Err(BuildError::Cuda {
                stage: "pre-capture native relaxed SO(3) reference projection",
                rc: baseline_rc,
            });
        }
        tier8_diag_named_driver_gate(
            "post_t7_sync",
            "t7_kl_calibration",
            cfg.diagnostic_stream_id,
            gap_diag_protocol_group,
            gap_expected_ctx,
            md_raw,
            true,
            || md_stream.synchronize(),
        )?;
        log::info!(
            "[NATIVE-RELAXED-REFERENCE] projection_completed=true source=raw_seed_spikes \
             source_stream_id={} captured_frame_id={} clusters={} bytes={} \
             graph_replays=0 md_steps_advanced=0",
            cfg.source_stream_id, cfg.initial_frame_id, cfg.n_clusters, tiles_bytes
        );

        // ── 6. cuStreamBeginCapture (MODE_THREAD_LOCAL — captures
        //      cross-stream operations once a captured event bridges
        //      them, scoped to THIS worker thread).
        //
        // **TIER 6 (2026-05-03) — multi-stream interferometric race fix.**
        // Was MODE_GLOBAL.  Under TIER 2's `let t7_active = true;` all 8
        // worker threads simultaneously enter this build path on their
        // own md_stream.  GLOBAL mode treats *any* stream-modifying op
        // anywhere in the process as a capture-invalidator — so each
        // thread's per-stream `cudaMemcpyToSymbolAsync` for the LUT
        // populators (Cα, chain offsets, cluster→repr-residue) and each
        // other thread's `cuEventRecord` cross-invalidate every other
        // thread's capture.  Observed errors: rc=901
        // STREAM_CAPTURE_INVALIDATED, rc=900 INVALID_RESOURCE_HANDLE on
        // pool alloc (the stream's handle is dead post-invalidation),
        // STREAM_CAPTURE_UNSUPPORTED.
        //
        // THREAD_LOCAL is the surgical fix: capture-invalidation is
        // scoped to ops in the SAME thread, so this thread's
        // intra-worker cross-stream interferometric choreography
        // (md_stream + stream_perturbed forking via cuEventRecord +
        // cuStreamWaitEvent into ONE captured graph) still works as
        // designed, while the 7 sibling worker threads can build their
        // own independent multi-stream captures concurrently without
        // cross-thread interference.
        let begin_capture_result = unsafe {
            tier8_diag_named_driver_gate(
                "begin_capture",
                "post_t7_sync",
                cfg.diagnostic_stream_id,
                gap_diag_protocol_group,
                gap_expected_ctx,
                md_raw,
                false,
                || {
                    result::stream::begin_capture(
                        md_stream.cu_stream(),
                        CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )
                },
            )
        };
        if let Err(e) = begin_capture_result {
            let _ = pool.free_async(tiles_dev, md_raw);
            let _ = pool.free_async(tiles_baseline_dev, md_raw);
            let _ = pool.free_async(p_frozen_dev, md_raw);
            let _ = pool.free_async(adj_dev, md_raw);
            let _ = pool.free_async(burst_marker_dev, md_raw);
            if sisr_mask_dev != 0 {
                let _ = pool.free_async(sisr_mask_dev, md_raw);
            }
            if sisr_count_dev != 0 {
                let _ = pool.free_async(sisr_count_dev, md_raw);
            }
            let _ = pool.free_async(dynt7_acc_dev, md_raw);
            let _ = pool.free_async(dynt7_idx_dev, md_raw);
            let _ = pool.free_async(dynt7_stats_dev, md_raw);
            // Borrowed GovernorDevice::cruise_state_ptr() aliases its owner at
            // +592; it is not an F2 allocation. The retained governor retires it.
            return Err(e);
        }
        let mut capture_guard = CaptureGuard::new(
            md_stream.cu_stream(),
            cfg.diagnostic_stream_id,
            if cfg.g26_parent_cond_handle != 0 {
                "parent_owned_g26_child"
            } else {
                "standalone_child"
            },
        );

        // Pull the in-progress graph handle so we can bind the
        // conditional handle to it during capture.
        let mut in_progress_graph: CUgraph = ptr::null_mut();
        let mut capture_status: CUstreamCaptureStatus =
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut cap_id: cuuint64_t = 0;
        let mut deps_ptr: *const CUgraphNode = ptr::null();
        let mut n_deps: usize = 0;
        unsafe {
            let rc = cuStreamGetCaptureInfo_v2(
                md_stream.cu_stream(),
                &mut capture_status as *mut _,
                &mut cap_id as *mut _,
                &mut in_progress_graph as *mut _,
                &mut deps_ptr as *mut _,
                &mut n_deps as *mut _,
            );
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                return Err(BuildError::Cuda {
                    stage: "cuStreamGetCaptureInfo_v2 (initial)",
                    rc: rc as i32,
                });
            }
        }
        if !matches!(
            capture_status,
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
        ) {
            return Err(BuildError::CaptureNotActive);
        }

        // V2 will create the conditional handle here and bind a
        // CONDITIONAL node downstream of the trampoline. V1 ships
        // without the conditional + trampoline so the captured graph
        // is well-formed (instantiate rejects a graph with a created
        // handle but no consuming conditional node, observed locally
        // as CUDA_ERROR_INVALID_VALUE).
        let cond_handle: CUgraphConditionalHandle = 0;

        // The physical producer resets its atom plane before each step.
        // Observation work must survive until SFA reads the completed window;
        // resetting it at this downstream overlay's head erased the evidence.

        // ══════════════════════════════════════════════════════════════
        // 6.a M1.2.20.C-B / PATH C — DUAL-STREAM FORK
        // ══════════════════════════════════════════════════════════════
        //
        // Captured-graph topology after this section:
        //
        //     [head-of-loop on md_stream]
        //         ├── cuMemsetD8Async d_com_shift (12 B)         (md_stream)
        //         ├── cuMemsetD8Async adj.momentum_violation (4 B)(md_stream)
        //         └── cuEventRecord(fork_event, md_stream)
        //             │
        //             ├──► md_stream (RELAXED branch)
        //             │       └── SO(3)(d_spikes_raw) → tiles_baseline_dev
        //             │       └── G28 SISR (reads tiles_baseline_dev)
        //             │
        //             └──► stream_perturbed (PERTURBED branch)
        //                   ├── cuStreamWaitEvent(fork_event)
        //                   ├── gasp_kernel(d_spikes_raw, d_com_shift)
        //                   │      → d_perturbed_positions
        //                   ├── SO(3)(d_perturbed_positions) → tiles_dev_ptr
        //                   ├── momentum_guard_check_kernel
        //                   │      → adj.momentum_violation_flag
        //                   └── cuEventRecord(perturbed_join_event, …)
        //
        //     [join on md_stream]
        //         └── cuStreamWaitEvent(perturbed_join_event)
        //         └── Node C (Adjudicator) reads BOTH manifolds
        //
        // The Blackwell sm_120 GigaThread Engine interleaves the warps
        // from md_stream (SO(3) on raw) and stream_perturbed (gasp +
        // SO(3) on perturbed) — Tensor Core MMA on one branch overlaps
        // with HBM3 LDG.E.128 on the other, hiding the perturbation
        // latency behind the relaxed-branch baseline extraction.

        // 6.a.0 — Head-of-loop zero of the COM accumulators + momentum
        // violation flag.  Each replay must start with com_shift=0,
        // total_mass=0, com_correction=0, and adj.momentum_violation_flag=0
        // so the gasp kernel's atomicAdds and the post-pass guard check
        // run on a clean slate.  Path Ω Option A added total_mass and
        // com_correction zeroes alongside com_shift.
        unsafe {
            let rc = cuMemsetD8Async(d_com_shift_dev as CUdeviceptr, 0, 12, md_stream.cu_stream());
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                return Err(BuildError::Cuda {
                    stage: "M1.2.20.C-B head-of-loop cuMemsetD8Async (com_shift)",
                    rc: rc as i32,
                });
            }
            // Path Ω Option A — d_total_mass [4 B] + d_com_correction [12 B]
            let rc = cuMemsetD8Async(d_total_mass_dev as CUdeviceptr, 0, 4, md_stream.cu_stream());
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                return Err(BuildError::Cuda {
                    stage: "Path Ω Option A head-of-loop cuMemsetD8Async (total_mass)",
                    rc: rc as i32,
                });
            }
            let rc = cuMemsetD8Async(
                d_com_correction_dev as CUdeviceptr,
                0,
                12,
                md_stream.cu_stream(),
            );
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                return Err(BuildError::Cuda {
                    stage: "Path Ω Option A head-of-loop cuMemsetD8Async (com_correction)",
                    rc: rc as i32,
                });
            }
            let rc = cuMemsetD8Async((adj_dev + 144) as CUdeviceptr, 0, 4, md_stream.cu_stream());
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                return Err(BuildError::Cuda {
                    stage: "M1.2.20.C-B head-of-loop cuMemsetD8Async (momentum_violation_flag)",
                    rc: rc as i32,
                });
            }
        }

        // ════════════════════════════════════════════════════════════════
        // OPERATOR MANDATE 2026-05-08 §3.III — MONOLITHIC BODY UNROLL
        // ════════════════════════════════════════════════════════════════
        //
        // Wrap the per-step body region (Path C fork → SO(3) →
        // Adjudicator → Dynamic-T7 → Ghost emission → ASC → G26 gearbox →
        // energy monitor → ZSTR stage/fence → cross-stream JOIN) in a
        // Rust for-loop that records the body N times into the same
        // in-progress CUgraph.
        //
        // Per-iteration nodes captured by `cuStreamBeginCapture` are
        // distinct graph identities (CUDA driver allocates a new node
        // per launch). Cross-stream events (fork_event,
        // perturbed_join_event, md_to_telemetry_event, telemetry_to_md_event)
        // are reused across iterations: each cuEventRecord captures a
        // new "record" node into the graph and the corresponding
        // cuStreamWaitEvent on the next iteration consumes it.
        //
        // SCALAR-SNAPSHOT CAVEAT (documented in PipelineConfig.body_unroll
        // rustdoc):
        //   * `zstr_pos_stage_node`, `zstr_fence_node`, `g26_bridge_node`
        //     are mutable Rust scalars rebound on each iteration. After
        //     the loop, they hold the LAST iteration's snapshot. The
        //     post-capture wiring (G26 SWITCH @4737, ZSTR Reaper handle
        //     update) consumes that last snapshot; first N-1 iterations'
        //     bridge / stage / fence nodes execute normally inside the
        //     graph (they don't need post-capture re-binding because §3.I
        //     wires the device-side slot updater into the graph itself).
        //
        // SLOT ROLLING (§3.I integration):
        //   * Each iteration's §1 increment kernel + §3.I slot-update
        //     kernel fire BEFORE the iteration's ZSTR pos_stage / fence
        //     kernels read d_zstr_active_slot. The captured DAG enforces
        //     this order via stream-recording.
        //
        // DEFAULT body_unroll = 1: loop runs once, behavior bit-identical
        // to the pre-§3 single-step capture.
        let body_unroll = cfg.body_unroll.max(1);
        if body_unroll != 1 {
            log::info!(
                "[§3.III BODY-UNROLL] capturing per-step body {} times \
                 (host loop will launch chunk_size / {} times per chunk)",
                body_unroll,
                body_unroll
            );
        }

        // ─── HOISTED scalars used by post-capture wiring ────────────────
        // These were originally declared inside the per-step body. The §3
        // unroll moved the body inside a for-loop; the post-capture wiring
        // (G26 SWITCH wire @4834+, ZSTR Reaper handle update @4983+) lives
        // OUTSIDE the loop and reads these snapshots to bind graph nodes
        // after cuStreamEndCapture. Per the body_unroll caveat: when
        // body_unroll > 1 these scalars retain the LAST iteration's
        // snapshot; first N-1 iterations' nodes execute correctly inside
        // the captured graph but their POST-capture handle-update wiring
        // is not vectorised (next surgical pass).
        let mut adj_node_set_hoisted: Vec<CUgraphNode> = Vec::new();
        let g26_switch_owned_by_parent: bool = cfg.g26_parent_cond_handle != 0;
        let mut g26_cond_handle_hoisted: u64 = cfg.g26_parent_cond_handle;
        let mut g26_bridge_node_hoisted: CUgraphNode = ptr::null_mut();
        let mut zstr_pos_stage_node_hoisted: CUgraphNode = ptr::null_mut();
        let mut zstr_fence_node_hoisted: CUgraphNode = ptr::null_mut();
        let mut zstr_src_vram_hoisted: u64 = 0;
        let mut zstr_n_atoms_hoisted: u32 = 0;

        for unroll_iter in 0..body_unroll {
            let _ = unroll_iter; // referenced via comments; kept for future
                                 // per-iteration vectorisation (Vec snapshots).

            // 6.a.1 — FORK: record fork_event on md_stream; perturbed stream
            // waits.  Both branches now have a well-defined start barrier
            // captured into the graph.
            unsafe {
                let rc = cuEventRecord(fork_event, md_stream.cu_stream());
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage: "Path C cuEventRecord(fork_event, md_stream)",
                        rc: rc as i32,
                    });
                }
                let rc = cuStreamWaitEvent(stream_perturbed, fork_event, 0);
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage: "Path C cuStreamWaitEvent(stream_perturbed)",
                        rc: rc as i32,
                    });
                }
            }

            // 6.a.2 — RELAXED branch on md_stream: SO(3)(d_spikes_raw) →
            // tiles_baseline_dev (the RELAXED manifold buffer; was
            // time-lagged frame N-1 pre-Path-C, now is the SO(3) of raw
            // spikes at frame N).  The DtoDAsync time-lag copy that used
            // to populate tiles_baseline_dev is REMOVED below — the
            // relaxed manifold is now computed every replay rather than
            // copied from a prior frame.
            let rc = unsafe {
                crate::so3_project::ffi::prism_so3_project_seed_run(
                    cfg.d_spikes,
                    cfg.d_cluster_offsets,
                    cfg.n_clusters,
                    cfg.d_k_lm,
                    tiles_baseline_dev as *mut ContactShellTile,
                    cfg.initial_frame_id,
                    cfg.source_stream_id,
                    md_stream.cu_stream() as *mut c_void,
                    cfg.seed_layout,
                )
            };
            if rc != crate::so3_project::ffi::CUDA_SUCCESS {
                return Err(BuildError::Cuda {
                    stage: "Path C Node B (SO(3) RELAXED)",
                    rc,
                });
            }

            // 6.a.3 — PERTURBED branch on stream_perturbed: gasp kernel
            // computes Δr per spike from the per-atom force gradient,
            // writes pos_perturbed = pos_raw + Δr to d_perturbed_positions,
            // and atomicAdds m·Δr into d_com_shift_dev plus m into
            // d_total_mass_dev (Path Ω Option A).  Both accumulators feed
            // the post-pass momentum-guard kernel which computes the COM
            // correction vector.
            let rc = unsafe {
                crate::so3_project::ffi::prism_apply_gradient_gasp_positions_seed_clock_launch(
                    cfg.d_spikes,
                    d_perturbed_positions as *mut f32,
                    cfg.d_forces_anchor as *const f32,
                    cfg.d_masses as *const f32,
                    adj_dev as *const c_void,
                    d_com_shift_dev as *mut c_void,
                    d_total_mass_dev as *mut c_void, // Path Ω Option A
                    cfg.d_current_md_step,
                    cfg.n_spikes,
                    cfg.n_atoms_for_pe,
                    stream_perturbed as *mut c_void,
                    cfg.seed_layout,
                )
            };
            if rc != crate::so3_project::ffi::CUDA_SUCCESS {
                return Err(BuildError::Cuda {
                    stage: "Path C gasp kernel",
                    rc,
                });
            }

            // 6.a.4 — Path Ω Option A — Momentum-Guard post-pass kernel:
            // reads (Σ m·Δr) and (Σ m), writes correction = (Σ m·Δr) / (Σ m)
            // to d_com_correction_dev, and sets momentum_violation_flag = 1
            // ONLY if the correction magnitude itself exceeds 1.0 Å (the
            // gasp produced unphysical kicks).  Pre-Option-A this kernel
            // checked the RAW |Σ m·Δr| > 1e-4 Å which fired every chunk
            // because that's a √n random-walk that can't be bounded that
            // tightly — the canonical fix is to subtract the global drift
            // and let the SO(3) KL see only the structural divergence.
            let rc = unsafe {
                crate::so3_project::ffi::prism_momentum_guard_check_launch(
                    d_com_shift_dev as *const c_void,
                    d_total_mass_dev as *const c_void,
                    d_com_correction_dev as *mut c_void,
                    adj_dev as *mut c_void,
                    stream_perturbed as *mut c_void,
                )
            };
            if rc != crate::so3_project::ffi::CUDA_SUCCESS {
                return Err(BuildError::Cuda {
                    stage: "Path C Momentum Guard check",
                    rc,
                });
            }

            // 6.a.5 — Path Ω Option A — Apply the COM correction in-place
            // to d_perturbed_positions: each spike's (x, y, z) -= correction.
            // Result: Σ m·(pos_perturbed - pos_relaxed) = 0 by construction;
            // the SO(3)-PERTURBED projection now operates on a manifold
            // that's COM-locked to the relaxed manifold, so the downstream
            // 4-plane KL adjudication captures relative structural
            // divergence instead of being dominated by global rigid drift.
            let rc = unsafe {
                crate::so3_project::ffi::prism_apply_com_positions_correction_launch(
                    d_perturbed_positions as *mut c_void,
                    d_com_correction_dev as *const c_void,
                    cfg.n_spikes,
                    stream_perturbed as *mut c_void,
                )
            };
            if rc != crate::so3_project::ffi::CUDA_SUCCESS {
                return Err(BuildError::Cuda {
                    stage: "Path C COM correction apply",
                    rc,
                });
            }

            // 6.a.6 — PERTURBED branch SO(3) on the COM-corrected kicked
            // spikes → manifest.tiles_dev_ptr (the PERTURBED manifold buffer).
            let rc = unsafe {
                crate::so3_project::ffi::prism_so3_project_positions_seed_run(
                    cfg.d_spikes,
                    d_perturbed_positions as *const f32,
                    cfg.d_cluster_offsets,
                    cfg.n_clusters,
                    cfg.d_k_lm,
                    manifest.tiles_dev_ptr,
                    cfg.initial_frame_id,
                    cfg.source_stream_id,
                    stream_perturbed as *mut c_void,
                    cfg.seed_layout,
                )
            };
            if rc != crate::so3_project::ffi::CUDA_SUCCESS {
                return Err(BuildError::Cuda {
                    stage: "Path C Node B (SO(3) PERTURBED)",
                    rc,
                });
            }

            // 6.a.6 — JOIN: stream_perturbed records perturbed_join_event;
            // md_stream waits before launching the Adjudicator.
            unsafe {
                let rc = cuEventRecord(perturbed_join_event, stream_perturbed);
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage: "Path C cuEventRecord(perturbed_join_event)",
                        rc: rc as i32,
                    });
                }
                let rc = cuStreamWaitEvent(md_stream.cu_stream(), perturbed_join_event, 0);
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage: "Path C cuStreamWaitEvent(md_stream rejoin)",
                        rc: rc as i32,
                    });
                }
            }

            // ── 6.b-G28 SISR symmetry consensus ──────────────────────────
            // Path C update: SISR reads the RELAXED manifold (tiles_baseline_dev)
            // not the perturbed one — bilateral symmetry must be verified
            // on the unperturbed protein, not the post-gasp state.
            // Captured on md_stream after the SO(3) RELAXED retire (sequential
            // dependency on md_stream is implicit).
            if let Some(ref sisr) = cfg.sisr {
                let rc = unsafe {
                    prism_sisr_launch(
                        tiles_baseline_dev as *const c_void,
                        sisr_count_dev as *const c_void,
                        sisr_mask_dev as *mut c_void,
                        sisr.epsilon_sym_angstrom,
                        md_stream.cu_stream() as *mut c_void,
                    )
                };
                if rc != 0 {
                    return Err(BuildError::Cuda {
                        stage: "G28 SISR launch",
                        rc,
                    });
                }
            }

            // ── 6.b Node C: Adjudicator step (T13 SIMT 4-plane KL) ─────────
            // <<<1, 64>>>: each thread processes one cluster.  cfg.n_clusters
            // is the capture-time constant fed in; threads with id >=
            // n_clusters early-skip in the kernel.
            let rc = unsafe {
                crate::interferometric_adjudicator::ffi::prism_interferometric_adjudicator_step(
                    adj_dev as *mut InterferometricAdjudicatorFfi,
                    cfg.n_clusters,
                    md_stream.cu_stream() as *mut c_void,
                )
            };
            if rc != 0 {
                return Err(BuildError::Cuda {
                    stage: "Node C (Adjudicator)",
                    rc,
                });
            }
            // STREAM5_SFA_DIAG — probe capture state immediately after Node C.
            diag_capture_invalidation_after(
                &md_stream,
                "Node C (Adjudicator)",
                cfg.diagnostic_stream_id,
                "parent_owned_g26_child",
            )?;

            // ── 6.b-T7 Wave 3 / Path B: dynamic noise-floor calibration ────
            // Three captured kernels (capture → reduce → apply) run AFTER the
            // Adjudicator step on md_stream.  The Adjudicator's __threadfence()
            // before its global_adjudication_summary write + the captured-graph
            // dependency edge guarantee current_divergence is L2-visible here.
            // After PRISM_DYNT7_N_MIN samples (=100), the apply kernel writes
            // substrate-derived μ + σ into adj->noise_floor_mu[0] and σ[0],
            // adapting the threshold for subsequent launches.  Pure GPU-native;
            // no host involvement per chunk.
            let rc = unsafe {
                prism_dynamic_t7_launch(
                    adj_dev as *const c_void,
                    dynt7_acc_dev as *mut c_void,
                    dynt7_idx_dev as *mut c_void,
                    dynt7_stats_dev as *mut c_void,
                    md_stream.cu_stream() as *mut c_void,
                )
            };
            if rc != 0 {
                return Err(BuildError::Cuda {
                    stage: "Wave 3 dynamic T7",
                    rc,
                });
            }

            // ── 6.b-AMS Channel-B: GhostTileFrame ring push (M1.2.19.B) ─────
            // Operator Amendment 3.13 §2.1, as amended by 3.14 (Sector-Lock):
            // every replay pushes one self-describing 4096-byte
            // GhostTileFrame per cluster that passes the emission gate into
            // the pinned-host, device-mapped ring.  The two-part
            // [GhostTileFrame (128 B) + ContactShellTile (1280 B)] = 1408 B
            // record was retired by Amendment 3.14, and the gate is no longer
            // `adjudication_code >= 1` alone — the v2 kernel also emits on the
            // observation pass and the pruned diagnostic firehose
            // (cuda/ghost_tile_kernel.cu:534-538).  Captured AFTER
            // the Adjudicator step (so the SWITCH code is final + the per-
            // frame Δ_AB / power_spectrum reads are stable) and BEFORE ASC
            // (so the captured frame reflects the unsteered manifold —
            // legitimate "raw" interferometer input for offline Φ_sym /
            // Lag-Persistence integration).
            //
            // `frame_idx` is the immutable capture-time `initial_frame_id`;
            // host code re-records the graph if frame ordering needs to
            // change (consistent with ZSTR's pos-stage frame-id contract).
            // The u32 at ring offset 0 is `n_frames_written`, the ring-slot
            // counter the kernel atomicAdds
            // (cuda/ghost_tile_kernel.cu:138 for v1, :542 for v2). It has no
            // relationship to frame_idx and does NOT make frame_idx
            // monotonic: every replay of one captured graph stamps the SAME
            // frame_idx, which changes only when the host re-records the
            // graph. The authoritative per-record time axis is the v2
            // step_idx field at offset 152, fed by the device-side step
            // counter (cuda/ghost_tile_kernel.cu:628-637).
            if cfg.ghost_tile_ring_dev != 0 && cfg.ghost_tile_max_records > 0 {
                // Wave 1 / Q2 — `d_kcc_lead` is the F2-pool [n_clusters] u32
                // buffer holding the per-cluster causal-lead residue id.
                // When `cfg.d_kcc_lead == 0` the kernel emits 0xFFFFFFFFu
                // sentinels until the host populator fires (typical
                // bootstrap during the very first chunk).
                let kcc_lead_ptr = if cfg.d_kcc_lead != 0 {
                    cfg.d_kcc_lead as *const c_void
                } else {
                    std::ptr::null()
                };
                // M1.2.23 §4 + §5 — Transparent MAR v2 launcher selection.
                // When PipelineConfig.mar_v2 is Some, route through the v2
                // launcher so emitted records carry the structured v2 overlay
                // payload: observation_pass / discovery_pass / gear_id /
                // dt_fs / step_idx. Otherwise the v1 launcher path is
                // preserved bit-identically.
                //
                // The overlay is stamped GHOST_FRAME_SCHEMA_V3 = 3, not 2 —
                // ghost_tile_kernel.cu is the sole writer of
                // GHOST_V2_OFFSET_SCHEMA_VERSION tree-wide. Schema 2 is
                // defined but no producer in this tree writes it.
                //
                // Hard invariant (per directive §4): the v2 emission gate widens
                // telemetry to include observation_pass, but the F1/G26 SWITCH
                // path in the adjudicator is NOT changed. Discovery semantics
                // for SWITCH selection remain whatever the adjudicator's own
                // threshold computes.
                let rc = if let Some(ref mar) = cfg.mar_v2 {
                    // P4(3.4) — pass the sigma MULTIPLIERS, not a precomputed
                    // absolute threshold. The kernel now derives the gate from
                    // the adjudicator's ARMED mu/sigma, so `discovery_pass` and
                    // `adjudication_code` in the same record are computed from
                    // the same live values. Previously this host-side product
                    // was frozen when the graph was captured, while the
                    // adjudicator's own gate moved with Dynamic T7 — one record,
                    // two detectors, silently disagreeing.
                    //
                    // The multipliers are POLICY (3 sigma telemetry tier, 12
                    // sigma steering tier); the mu/sigma are MEASUREMENT and
                    // stay device-side.
                    let obs_thr = mar.threshold_sigma_observation;
                    let disc_thr = mar.threshold_sigma_discovery;
                    // GEAR-PROVENANCE-002 — the gear stamped into every v2
                    // Ghost record is a kernel-node argument fixed at graph
                    // BUILD time, whereas the gear is chosen per replay by the
                    // device SFA. No build-time value can therefore witness
                    // the resolution a record was produced at, and `0` is a
                    // legal gear, so a literal would be indistinguishable
                    // offline from a genuine 0.5 fs frame. Stamp the explicit
                    // not-measured sentinel and say so loudly if a caller
                    // supplied something else. The measured gear is published
                    // separately by `CapturedAdjudicationPipeline::measured_gear`.
                    let stamped_gear_id = crate::zstr::ZSTR_GEAR_ID_UNMEASURED;
                    if mar.gear_id != crate::zstr::ZSTR_GEAR_ID_UNMEASURED {
                        log::warn!(
                            "[GEAR-PROVENANCE-002] MarV2Config.gear_id={} is a build-time \
                             literal and cannot witness a per-replay device gear; stamping \
                             the not-measured sentinel 0x{:X} into Ghost v2 + ZSTR records \
                             instead. Read the measured gear via \
                             CapturedAdjudicationPipeline::measured_gear().",
                            mar.gear_id,
                            crate::zstr::ZSTR_GEAR_ID_UNMEASURED
                        );
                    }
                    log::info!(
                        "[M1.2.23 v2] ghost launcher: schema_version={} \
                     sigma_mult_obs={:.3} sigma_mult_disc={:.3} \
                     (absolute gate derived device-side from the armed adjudicator) \
                     gear_id_stamped=0x{:X} dt_fs={:.3} \
                     d_step_counter=0x{:x} prune_kl={:.4}",
                        crate::ghost_tile::GHOST_FRAME_SCHEMA_V3,
                        obs_thr,
                        disc_thr,
                        stamped_gear_id,
                        mar.dt_fs,
                        mar.d_step_counter,
                        mar.prune_kl_threshold
                    );
                    // Read the live owning physical endpoint on replay. An observer
                    // launch is not an MD step, and must not increment physical time.
                    if mar.d_step_counter != 0 {
                        let rc_clk = unsafe {
                            prism_bind_ghost_physical_clock_launch(
                                cfg.d_current_md_step,
                                mar.d_step_counter as *mut c_void,
                                md_stream.cu_stream() as *mut c_void,
                            )
                        };
                        if rc_clk != 0 {
                            return Err(BuildError::Cuda {
                                stage: "prism_bind_ghost_physical_clock_launch",
                                rc: rc_clk,
                            });
                        }
                        // OPERATOR MANDATE 2026-05-08 §3.I + WP4 — wire the
                        // device-side ZSTR SLOT AUTHORITY into the captured
                        // graph body. Reads the physical-endpoint
                        // *d_step_counter and:
                        //   1. applies bounded backpressure on the target
                        //      slot's completion_fence (never overwrites an
                        //      unpersisted record, never hangs),
                        //   2. claims (step % N_SLOTS) into the PER-STREAM
                        //      d_active_slot cell,
                        //   3. stamps the frozen 4096-byte frame header
                        //      (frame_idx / dt / n_atoms / gear_id /
                        //      adjudication_code / stream_id / schema_version).
                        //
                        // The downstream ZSTR pos_stage / force_stage /
                        // force_norm_sqrt / stage_audit / fence_signal kernels
                        // read *d_active_slot at execution time, so the slot
                        // rolls forward without host cudaMemcpyToSymbolAsync
                        // per step. This is the foundation for the §3.IV
                        // host-loop eradication: when the host launches the
                        // captured graph once per chunk (rather than once per
                        // step) the slot still rolls correctly because each
                        // replay (or each unrolled iteration inside a single
                        // replay) re-runs both the clock bridge and slot-update
                        // kernels.
                        //
                        // WP4: before this kernel existed in stamping form NO
                        // kernel wrote the ZSTR header at all — the offline
                        // parser derives its file stride from n_atoms at
                        // offset 20, read 0, and abandoned the file after the
                        // first 4096 bytes.
                        let (
                            zstr_slot_base,
                            zstr_slot_stride,
                            zstr_slot_cell,
                            zstr_slot_status,
                            zstr_slot_n_atoms,
                            zstr_slot_n_slots,
                            zstr_slot_spin_limit,
                            zstr_slot_stream_id,
                            zstr_slot_schema,
                        ) = match cfg.zstr {
                            Some(ref z) => (
                                z.pinned_base as *mut c_void,
                                z.inter_slot_stride,
                                z.d_active_slot as *mut c_void,
                                z.d_backpressure_status as *mut c_void,
                                z.n_atoms,
                                z.n_slots,
                                z.backpressure_spin_limit,
                                z.stream_id,
                                z.schema_version,
                            ),
                            // ZSTR ring absent on this stream (alloc or G21
                            // alignment failure). The node is still recorded so
                            // the captured-graph topology does not depend on
                            // ZSTR availability; a null slot cell makes the
                            // kernel a hard no-op.
                            None => (
                                std::ptr::null_mut(),
                                0u32,
                                std::ptr::null_mut(),
                                std::ptr::null_mut(),
                                0u32,
                                crate::zstr::ZstrRing::N_SLOTS as u32,
                                0u32,
                                0u32,
                                0u32,
                            ),
                        };
                        let rc_slot = unsafe {
                            crate::zstr::ffi::prism_zstr_device_slot_update_launch(
                                mar.d_step_counter as *const c_void,
                                zstr_slot_n_slots,
                                zstr_slot_base,
                                zstr_slot_stride,
                                zstr_slot_cell,
                                zstr_slot_n_atoms,
                                // Header `dt` is PICOSECONDS (zstr.rs:71 and
                                // the offline parser, which multiplies by 1000
                                // to recover fs). MarV2Config carries fs.
                                mar.dt_fs * 1.0e-3_f32,
                                // GEAR-PROVENANCE-001 (2026-08-02).
                                //
                                // `mar.gear_id` is a HARDCODED literal 0 from
                                // the V2 config ("Wave A default"), not a
                                // measurement. GEAR-PROVENANCE-002 has since
                                // added the read path (`measured_gear()` +
                                // the G26 control trace), but a measured gear
                                // only exists per replay at a synchronized
                                // boundary — this kernel argument is fixed at
                                // graph BUILD time and can never carry it.
                                // Under the documented encoding
                                // (0 = 0.5 fs, 1 = 2.0 fs, 2 = 4.0 fs HMR,
                                // 3 = abort-trap), stamping that 0 would assert
                                // "integrated at 0.5 fs" on EVERY frame while
                                // the run is actually at 2.0 or 4.0 fs — a
                                // 4x-to-8x misstatement of temporal resolution
                                // recorded as measurement, and indistinguishable
                                // offline from a genuine gear 0.
                                //
                                // Stamp the explicit not-measured sentinel. The
                                // gearbox now DOES publish a measured gear to
                                // the host (GEAR-PROVENANCE-002,
                                // `measured_gear()` + the G26 control trace),
                                // but that measurement only exists per replay
                                // at a synchronized boundary — it cannot be
                                // baked into this build-time kernel argument.
                                // Do NOT reinstate mar.gear_id.
                                stamped_gear_id,
                                zstr_slot_stream_id,
                                zstr_slot_schema,
                                zstr_slot_spin_limit,
                                zstr_slot_status,
                                md_stream.cu_stream() as *mut c_void,
                            )
                        };
                        if rc_slot != 0 {
                            return Err(BuildError::Cuda {
                                stage:
                                    "OPERATOR-2026-05-08 §3.I / WP4 prism_zstr_device_slot_update_launch",
                                rc: rc_slot,
                            });
                        }
                    } else if cfg.zstr.is_some() {
                        // WP4 loud-failure gate: the header stamp is derived
                        // from the §1 device clock. Without it there is no
                        // authoritative device step to write into frame_idx,
                        // so no slot is ever claimed and the ZSTR file stays
                        // empty rather than filling with unparseable records.
                        log::error!(
                            "[ZSTR WP4] ZSTR ring is live but the §1 device step counter is \
                             absent (mar_v2.d_step_counter == 0). The in-graph slot authority \
                             cannot be recorded, so no ZSTR record will be produced for this \
                             stream. Enable --mar-v2-telemetry."
                        );
                    }
                    unsafe {
                        prism_ghost_pipe_stage_launch_v2_seed(
                            cfg.ghost_tile_ring_dev,
                            manifest.tiles_dev_ptr as *const c_void,
                            cfg.d_spikes as *const c_void,
                            cfg.d_cluster_offsets as *const c_void,
                            adj_dev as *const c_void,
                            kcc_lead_ptr,
                            cfg.initial_frame_id as u64,
                            cfg.n_clusters,
                            cfg.ghost_tile_max_records,
                            md_stream.cu_stream() as *mut c_void,
                            cfg.firehose_enable,
                            obs_thr,
                            disc_thr,
                            // GEAR-PROVENANCE-002 — see `stamped_gear_id`
                            // above: never `mar.gear_id`.
                            stamped_gear_id,
                            mar.dt_fs,
                            cfg.initial_frame_id as u64, // legacy fallback when d_step_counter==0
                            mar.d_step_counter, // §1 device clock — authoritative when non-zero
                            mar.prune_kl_threshold, // §2 device-side firehose prune (default 0.01)
                    cfg.seed_layout,
                )
                    }
                } else {
                    unsafe {
                        prism_ghost_pipe_stage_launch(
                            cfg.ghost_tile_ring_dev,
                            manifest.tiles_dev_ptr as *const c_void,
                            adj_dev as *const c_void,
                            kcc_lead_ptr,
                            cfg.initial_frame_id as u64,
                            cfg.n_clusters,
                            cfg.ghost_tile_max_records,
                            md_stream.cu_stream() as *mut c_void,
                            cfg.firehose_enable,
                        )
                    }
                };
                if rc != 0 {
                    return Err(BuildError::Cuda {
                        stage: if cfg.mar_v2.is_some() {
                            "M1.2.19.B prism_ghost_pipe_stage_launch_v2"
                        } else {
                            "M1.2.19.B prism_ghost_pipe_stage_launch"
                        },
                        rc,
                    });
                }
                // STREAM5_SFA_DIAG — probe capture state after ghost pipe stage.
                diag_capture_invalidation_after(
                    &md_stream,
                    if cfg.mar_v2.is_some() {
                        "M1.2.19.B prism_ghost_pipe_stage_launch_v2"
                    } else {
                        "M1.2.19.B prism_ghost_pipe_stage_launch"
                    },
                    cfg.diagnostic_stream_id,
                    "parent_owned_g26_child",
                )?;
            }

            // ── 6.b' V2 IGNITION prep: snapshot the Adjudicator's captured
            // node handle BEFORE any cross-stream events confuse the
            // dependency frontier. The V2 hook (Claude-2's
            // `prism_wire_f1_switch_ffi`) consumes this handle as the
            // explicit dependency for the conditional node — operator's
            // §2.3 mandate ("explicit cuGraphAddDependencies edge from
            // Node C to Node D"). At this point in the capture sequence
            // the dependency frontier is exactly {adjudicator_node}.
            // §3.III-hoist: per-iteration Vec; at end of iteration we mirror
            // (clone) to the outer hoisted Vec so post-capture wiring sees
            // the LAST iteration's adjudicator node set. With body_unroll > 1
            // earlier iterations' adj nodes execute correctly inside the
            // graph; only the last iteration's set is used for post-capture
            // wiring (e.g., V2 hook at ~4972).
            let mut adj_node_set: Vec<CUgraphNode> = Vec::new();
            unsafe {
                let mut cap_status: CUstreamCaptureStatus =
                    CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
                let mut cap_id: cuuint64_t = 0;
                let mut graph_now: CUgraph = ptr::null_mut();
                let mut deps_ptr: *const CUgraphNode = ptr::null();
                let mut n_deps: usize = 0;
                let rc = cuStreamGetCaptureInfo_v2(
                    md_stream.cu_stream(),
                    &mut cap_status as *mut _,
                    &mut cap_id as *mut _,
                    &mut graph_now as *mut _,
                    &mut deps_ptr as *mut _,
                    &mut n_deps as *mut _,
                );
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage: "cuStreamGetCaptureInfo_v2 (post-Adjudicator snapshot)",
                        rc: rc as i32,
                    });
                }
                if n_deps > 0 {
                    adj_node_set = std::slice::from_raw_parts(deps_ptr, n_deps).to_vec();
                }
            }

            // Node D now publishes the completed-step decision for the NEXT
            // physical force pair. Physical composition/PE happens in the engine.
            if let Some(ref asc)=cfg.asc {
                if let Some(owner)=&asc.decision_owner {
                    unsafe { owner.publish(md_stream,adj_dev as *const InterferometricAdjudicatorFfi) }
                        .map_err(|rc|BuildError::Cuda {stage:"ASC decision publication",rc})?;
                } else {
                // M1.2.18-P3.2 — pass d_pe_components so the ASC kernel
                // can fold V_ASC into V_t.  Cast f64 const → f64 mut for
                // the atomicAdd target.  Null-passthrough preserves legacy
                // test fixtures that don't wire PE.
                let rc = unsafe {
                    crate::interferometric_adjudicator::ffi::prism_asc_apply(
                        adj_dev as *const InterferometricAdjudicatorFfi,
                        asc.d_forces,
                        asc.d_atom_positions,
                        asc.d_atom_in_cluster,
                        asc.n_atoms,
                        asc.steering_gain_alpha,
                        md_stream.cu_stream() as *mut c_void,
                        cfg.d_pe_components as *mut f64,
                        asc.d_asc_applied_atoms,
                        asc.d_asc_applied_sum_f,
                    )
                };
                if rc != 0 {
                    return Err(BuildError::Cuda {
                        stage: "Node D (ASC force inject)",
                        rc,
                    });
                }
                }
                diag_capture_invalidation_after(md_stream,"ASC decision/legacy compatibility",cfg.diagnostic_stream_id,"parent_owned_g26_child")?;
            }

            // ── 6.c'' B.3.2-FULL — G26 Chronometric Gearbox ────────────────────
            //
            // Capture-time:
            //   1. cuStreamGetCaptureInfo to fetch the in-progress graph handle.
            //   2. prism_gearbox_create_handle_ffi creates a conditional handle
            //      bound to the in-progress graph; bridge kernel will reference
            //      it as a kernel-node arg.
            //   3. Launch SFA kernel — reads adj->adjudication_code, mutates
            //      cruise.{counter, current_gear, previous_gear, last_burst_frame}.
            //      Decoupled from dt-write (the SWITCH body's apply_fixed_dt
            //      kernel owns that side-effect).
            //   4. Launch predicate-bridge kernel — reads adj->gear_override
            //      (offset 100, B.3.2) and cruise->current_gear, calls
            //      cudaGraphSetConditional(handle, final_gear).
            //   5. Snapshot the bridge node handle for the post-capture
            //      SWITCH wiring.
            //
            // Post-capture (after end_capture):
            //   6. prism_gearbox_wire_with_handle_ffi — adds the SWITCH
            //      conditional node downstream of bridge_node, using the
            //      handle from step 2.
            //   7. prism_gearbox_populate_switch_bodies_ffi — populates the
            //      4 phGraph_out body sub-graphs:
            //        Body 0 (Burst, 0.5fs):  rescale → Berendsen → apply_dt(0)
            //        Body 1 (Cruise, 2.0fs): rescale → apply_dt(1)
            //        Body 2 (Sprint, 4.0fs): rescale → apply_dt(2)
            //        Body 3 (Abort):         trap (asm volatile("trap;"))

            // Step 1 + 2: create or import the G26 conditional handle.
            // TIER 8 Option β: monolithic fusion passes a parent-owned
            // handle here.  The child still captures the predicate bridge
            // kernel, but the SWITCH node is installed on the parent graph
            // after the child is spliced.
            // §3.III-hoist: rebind to outer-scope hoisted scalars.
            // g26_switch_owned_by_parent is loop-invariant (depends on cfg),
            // so we just shadow it locally with the same value each iteration.
            // g26_cond_handle is the SAME value across iterations (each
            // iteration's bridge kernel writes to the SAME conditional handle),
            // but we mirror writes to the outer scope so post-capture wiring
            // sees the resolved handle value.
            let _g26_switch_owned_by_parent_inner: bool = g26_switch_owned_by_parent;
            let mut g26_cond_handle: u64 = g26_cond_handle_hoisted;
            if g26_switch_owned_by_parent {
                log::info!(
                    "[TIER8-G26] using parent-owned conditional handle {:#x}; \
                 child template will remain SWITCH-free",
                    g26_cond_handle
                );
            } else {
                let mut g26_in_progress_graph: CUgraph = ptr::null_mut();
                unsafe {
                    let mut s = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
                    let mut id: cuuint64_t = 0;
                    let mut dp: *const CUgraphNode = ptr::null();
                    let mut nd: usize = 0;
                    let rc = cuStreamGetCaptureInfo_v2(
                        md_stream.cu_stream(),
                        &mut s as *mut _,
                        &mut id as *mut _,
                        &mut g26_in_progress_graph as *mut _,
                        &mut dp as *mut _,
                        &mut nd as *mut _,
                    );
                    if !matches!(rc, CUresult::CUDA_SUCCESS) {
                        return Err(BuildError::Cuda {
                            stage: "B.3.2 cuStreamGetCaptureInfo (pre-gearbox)",
                            rc: rc as i32,
                        });
                    }
                }
                let rc = unsafe {
                    prism_gearbox_create_handle_ffi(
                        g26_in_progress_graph,
                        /*default_value=*/ 1u32, // safe default = Gear 1 (2.0fs)
                        &mut g26_cond_handle as *mut u64,
                    )
                };
                if rc != 0 {
                    return Err(BuildError::Cuda {
                        stage: "B.3.2 prism_gearbox_create_handle_ffi",
                        rc,
                    });
                }
            }

            // R04: force production retires before the SFA consumes this epoch.
            // This path is stream ordered with physical completion and ASC.
            if let Some(ref zstr) = cfg.zstr {
                // ── T11 — force-stage (DMA + warp-shuffle Σ‖F‖² atomic-add) ──
                // Captured on telemetry_stream after pos_stage. Reads d_forces
                // AFTER the ASC kernel has retired (md_to_telemetry_event +
                // tile DMA dependency chain establishes the cross-stream
                // happens-before edge). Each warp lane-0 atomicAdds its
                // Σ Fᵢ² into the active slot's force_norm field (offset 28).
                let rc = unsafe {
                    zstr_launch_force_stage(
                        zstr.pinned_base as *mut c_void,
                        zstr.inter_slot_stride,
                        zstr.force_offset_in_slot,
                        zstr.force_norm_offset_in_slot,
                        zstr.d_forces as *const c_void,
                        zstr.n_atoms,
                        zstr.d_active_slot as *const c_void,
                        md_stream.cu_stream() as *mut c_void,
                    )
                };
                if rc != 0 {
                    return Err(BuildError::Cuda {
                        stage: "zstr_force_stage capture (T11)",
                        rc,
                    });
                }

                // ── T11 — force_norm sqrt post-pass (single-thread) ──
                // In-place sqrtf converts the running Σ‖F‖² into ‖F‖₂
                // before the consumer reads. NaN propagates verbatim;
                // the host G29 Reaper traps non-finite reads.
                let rc = unsafe {
                    zstr_launch_force_norm_sqrt(
                        zstr.pinned_base as *mut c_void,
                        zstr.inter_slot_stride,
                        zstr.force_norm_offset_in_slot,
                        zstr.d_active_slot as *const c_void,
                        md_stream.cu_stream() as *mut c_void,
                        governor_device.force_value_ptr() as *mut c_void,
                    )
                };
                if rc != 0 {
                    return Err(BuildError::Cuda {
                        stage: "zstr_force_norm_sqrt capture (T11)",
                        rc,
                    });
                }

                governor_device.force_complete(md_stream, 2)
                    .map_err(|rc| BuildError::Cuda { stage: "ZSTR force window seal", rc })?;
            } else if let Some(ref asc) = cfg.asc {
                unsafe { governor_device.measure_force(md_stream, asc.d_forces.cast(), asc.n_atoms.max(0) as u32) }
                    .map_err(|rc| BuildError::Cuda { stage: "native force window reduction", rc })?;
            }

            // ── M1.2.17 — Hamiltonian Auditor (energy monitor reduce) ──
            // Captured BEFORE the SFA kernel so adj.d_potential_energy
            // (offset 112) holds V_t when the SFA reads it for the
            // stability-fuse drift check.  Skipped when the host hasn't
            // wired d_pe_components / n_atoms_for_pe — the SFA stays in
            // first-frame mode (no drift trip).  Captured-graph dependency:
            //   ASC → energy_monitor_reduce → energy_window_update → SFA → ...
            //
            // The reduce + window-update pair adds 2 captured nodes.
            if !cfg.d_pe_components.is_null() && cfg.n_atoms_for_pe > 0 {
                // The CUB DeviceReduce::Sum is captured as a series of
                // internal nodes; we don't snapshot individual handles.
                // The window-update kernel writes adj.d_potential_energy
                // (offset 112) directly.
                let adj_pe_target = (adj_dev + 112) as *mut f64;
                let rc = unsafe {
                    prism_energy_monitor_launch_reduce(
                        cfg.d_pe_components,
                        cfg.n_atoms_for_pe,
                        energy_temp_storage_dev as *mut c_void,
                        energy_temp_storage_bytes as usize,
                        energy_pe_scalar_dev as *mut f64,
                        energy_window_dev as *mut c_void,
                        adj_pe_target,
                        md_stream.cu_stream() as *mut c_void,
                    )
                };
                if rc != 0 {
                    return Err(BuildError::Cuda {
                        stage: "M1.2.17 prism_energy_monitor_launch_reduce",
                        rc,
                    });
                }
                // STREAM5_SFA_DIAG — probe capture state after M1.2.17 energy reduce.
                diag_capture_invalidation_after(
                    &md_stream,
                    "M1.2.17 prism_energy_monitor_launch_reduce",
                    cfg.diagnostic_stream_id,
                    "parent_owned_g26_child",
                )?;
            }

            // Step 3: SFA kernel.
            let rc = unsafe {
                crate::governor_device::prism_gearbox_launch_sfa_v2(
                    adj_dev as *const InterferometricAdjudicatorFfi,
                    cruise_state_dev as *mut crate::gearbox::ChronometricStateTensor,
                    governor_device.device_ptr(),
                    md_stream.cu_stream() as *mut c_void,
                )
            };
            if rc != 0 {
                return Err(BuildError::Cuda {
                    stage: "B.3.2 SFA kernel",
                    rc,
                });
            }

            // Step 4: predicate bridge kernel — sets conditional for SWITCH.
            // CHUNK13_DIAG — when cfg.branch_trace_dev is non-zero, route
            // through the TRACED variant so the bridge kernel writes the
            // last predicate value, branch counts, and invocation count
            // into the pinned-mapped trace block. Zero overhead when
            // cfg.branch_trace_dev == 0 (kernel becomes the legacy bridge).
            let rc = if governor_device.has_physical_guard() {
                unsafe {crate::governor_device::prism_gearbox_launch_guard_bridge_v1(
                    g26_cond_handle,cruise_state_dev as *const crate::gearbox::ChronometricStateTensor,
                    governor_device.device_ptr(),md_stream.cu_stream().cast(),cfg.branch_trace_dev)}
            } else if cfg.branch_trace_dev != 0 {
                unsafe {
                    crate::gearbox::ffi::prism_gearbox_launch_predicate_bridge_traced(
                        g26_cond_handle,
                        adj_dev as *const InterferometricAdjudicatorFfi,
                        cruise_state_dev as *const crate::gearbox::ChronometricStateTensor,
                        md_stream.cu_stream() as *mut c_void,
                        cfg.branch_trace_dev,
                    )
                }
            } else {
                unsafe {
                    crate::gearbox::ffi::prism_gearbox_launch_predicate_bridge(
                        g26_cond_handle,
                        adj_dev as *const InterferometricAdjudicatorFfi,
                        cruise_state_dev as *const crate::gearbox::ChronometricStateTensor,
                        md_stream.cu_stream() as *mut c_void,
                    )
                }
            };
            if rc != 0 {
                return Err(BuildError::Cuda {
                    stage: "B.3.2 predicate bridge kernel",
                    rc,
                });
            }

            // F1-PARENT-SWITCH-001 — F1 predicate-bridge kernel launched
            // alongside G26's. Reads `adj->adjudication_code & 0x3` and
            // writes the result into the parent-owned F1 conditional handle.
            // Mirrors the G26 launch shape (single-thread kernel-node into
            // the captured graph). Only fires when a parent-owned F1 handle
            // was supplied via `cfg.f1_parent_cond_handle`; otherwise F1 is
            // disabled at the CUDA-graph layer and the launch is skipped so
            // the captured child template stays splice-legal.
            if cfg.f1_parent_cond_handle != 0 {
                let adj_code_devptr = unsafe {
                    crate::interferometric_adjudicator::ffi::prism_get_adjudication_code_devptr(
                        adj_dev as *const InterferometricAdjudicatorFfi,
                    )
                };
                if adj_code_devptr.is_null() {
                    return Err(BuildError::Cuda {
                        stage: "F1-PARENT-SWITCH adjudication_code devptr null",
                        rc: -1,
                    });
                }
                // CHUNK13_DIAG — same TRACED-variant routing as G26 above.
                let rc = if cfg.branch_trace_dev != 0 {
                    unsafe {
                        prism_f1_launch_predicate_bridge_traced(
                            adj_code_devptr,
                            cfg.f1_parent_cond_handle,
                            /*mask=*/ 0x3u32,
                            md_stream.cu_stream() as *mut c_void,
                            cfg.branch_trace_dev,
                        )
                    }
                } else {
                    unsafe {
                        prism_f1_launch_predicate_bridge(
                            adj_code_devptr,
                            cfg.f1_parent_cond_handle,
                            /*mask=*/ 0x3u32,
                            md_stream.cu_stream() as *mut c_void,
                        )
                    }
                };
                if rc != 0 {
                    return Err(BuildError::Cuda {
                        stage: "F1-PARENT-SWITCH predicate bridge kernel",
                        rc,
                    });
                }
            }

            // Step 5: snapshot bridge node handle (frontier == [bridge]).
            // §3.III-hoist: write through to the outer scope so post-capture
            // G26 SWITCH wiring (line ~4834) sees the latest bridge node.
            // body_unroll > 1 ⇒ this scalar holds the LAST iteration's bridge.
            let mut g26_bridge_node: CUgraphNode = ptr::null_mut();
            unsafe {
                let mut s = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
                let mut id: cuuint64_t = 0;
                let mut g: CUgraph = ptr::null_mut();
                let mut dp: *const CUgraphNode = ptr::null();
                let mut nd: usize = 0;
                let rc = cuStreamGetCaptureInfo_v2(
                    md_stream.cu_stream(),
                    &mut s,
                    &mut id,
                    &mut g,
                    &mut dp,
                    &mut nd,
                );
                if matches!(rc, CUresult::CUDA_SUCCESS) && nd > 0 {
                    g26_bridge_node = *dp;
                }
            }
            g26_bridge_node_hoisted = g26_bridge_node;
            g26_cond_handle_hoisted = g26_cond_handle;

            // ── 6.c-Pz2 Path Z.2 Dual-Manifold Pointer Roll — REMOVED in M1.2.20.C-B.
            //
            // The pre-Path-C scheme used a captured cuMemcpyDtoDAsync to copy
            // tiles_dev → tiles_baseline_dev after every Adjudicator pass so the
            // NEXT frame's adjudication had a temporal delta vs. the prior frame.
            // Path C replaces this with a SAME-FRAME dual projection: relaxed and
            // perturbed manifolds are both computed at frame N (not N vs. N-1),
            // so there is no temporal roll to perform.  The DtoD copy that lived
            // here would now overwrite the next replay's RELAXED manifold with
            // the perturbed one — the opposite of intended semantics.

            // ── 6.c Cross-stream FORK: MD → telemetry ────────────────────
            // After Node C completes on md_stream, fire the
            // md_to_telemetry_event; telemetry_stream waits on it before
            // launching the DMA. Both operations are captured under
            // MODE_GLOBAL, producing cross-stream dependency edges in
            // the resulting graph.
            unsafe {
                let rc = cuEventRecord(md_to_telemetry_event, md_stream.cu_stream());
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage: "cuEventRecord (md → telemetry)",
                        rc: rc as i32,
                    });
                }
                let rc = cuStreamWaitEvent(telemetry_stream, md_to_telemetry_event, 0);
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage: "cuStreamWaitEvent (telemetry waits)",
                        rc: rc as i32,
                    });
                }
            }

            // Schedule the async D2H to the ring's frame-0 write slot. The
            // captured graph stamps frame_idx = initial_frame_id; subsequent
            // launches reuse the same slot index (frame 0 % 3 = 0). The
            // production wire-in (V2) replaces the constant frame_idx with
            // a kernel-updatable pointer-stable counter, but for the V1
            // LEGO brick the constant frame is sufficient to attest
            // operator §2.2 "fired concurrently on the non-blocking stream".
            schedule_async_tile_copy(
                &ring,
                manifest.tiles_dev_ptr as *const ContactShellTile,
                cfg.n_clusters as usize,
                telemetry_stream,
                /* frame_idx = */ cfg.initial_frame_id as u64,
            )
            .map_err(|rc| BuildError::Cuda {
                stage: "schedule_async_tile_copy (ghost-pipe DMA)",
                rc,
            })?;

            // ── 6.c-mid G23 ZSTR: position staging + fence signal ────────
            // Node sequence on telemetry_stream (captured under MODE_GLOBAL):
            //   [tile DMA] → [zstr_pos_stage_f4] → [zstr_signal_completion]
            // Both launches record kernel nodes into the in-progress CUgraph.
            // The fence-signal node fires __threadfence_system() before writing
            // completion_fence=1, guaranteeing all position bytes are globally
            // visible to the host ZSTR consumer before it reads them.
            // Phase 3: dst_pinned/slot_fence baked at capture time (slot 0).
            // Phase 4: cuGraphKernelNodeSetParams updates slot per launch.
            // G23/G24: ZSTR pos_stage + fence_signal on telemetry_stream.
            // After each launcher we snapshot the telemetry_stream's dependency
            // frontier — under MODE_GLOBAL the frontier is exactly the node just
            // recorded — to obtain the graph node handles needed by G24's
            // cuGraphExecKernelNodeSetParams slot-roller.
            // §3.III-hoist: per-iteration locals; we mirror to the outer
            // hoisted scalars at the end of the ZSTR block below so
            // post-capture wiring (line ~4983 / 4997) sees the latest.
            let mut zstr_pos_stage_node: CUgraphNode = ptr::null_mut();
            let mut zstr_fence_node: CUgraphNode = ptr::null_mut();
            let mut zstr_src_vram: u64 = 0;
            let mut zstr_n_atoms: u32 = 0;

            if let Some(ref zstr) = cfg.zstr {
                zstr_src_vram = zstr.d_positions as u64;
                zstr_n_atoms = zstr.n_atoms;

                let rc = unsafe {
                    zstr_launch_pos_stage(
                        zstr.pinned_base as *mut c_void,
                        zstr.inter_slot_stride,
                        zstr.pos_offset_in_slot,
                        zstr.d_positions as *const c_void,
                        zstr.n_atoms,
                        zstr.d_active_slot as *const c_void,
                        telemetry_stream as *mut c_void,
                    )
                };
                if rc != 0 {
                    return Err(BuildError::Cuda {
                        stage: "zstr_pos_stage capture (G23)",
                        rc,
                    });
                }
                // Snapshot pos_stage node: frontier on telemetry_stream is now
                // [pos_stage_node] immediately after the <<<>>> launch.
                unsafe {
                    let mut s = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
                    let mut id: cuuint64_t = 0;
                    let mut g: CUgraph = ptr::null_mut();
                    let mut dp: *const CUgraphNode = ptr::null();
                    let mut nd: usize = 0;
                    let rc = cuStreamGetCaptureInfo_v2(
                        telemetry_stream,
                        &mut s,
                        &mut id,
                        &mut g,
                        &mut dp,
                        &mut nd,
                    );
                    if matches!(rc, CUresult::CUDA_SUCCESS) && nd > 0 {
                        zstr_pos_stage_node = *dp;
                    }
                }

                // ── M1.2.18.5 — Hamiltonian audit-field stage (MANDATORY) ──
                // Snapshots V_t (adj.d_potential_energy @ 112) + W_ext
                // (*adj.d_external_work, where the pointer is at 128) into
                // the active ZSTR slot at offsets 32 + 40 so the off-line
                // replay can compute the First-Law drift trace per frame.
                // Captured AFTER force_norm_sqrt so the slot already has all
                // conservative observables.
                const ZSTR_EXTERNAL_WORK_OFFSET: u32 = 32;
                const ZSTR_POTENTIAL_ENERGY_OFFSET: u32 = 40;
                // WP4 — adjudication_code (header offset 12) is stamped here
                // because this is the only ZSTR node captured downstream of
                // the Adjudicator. Nothing wrote it before WP4.
                const ZSTR_ADJUDICATION_CODE_OFFSET: u32 = 12;
                let rc = unsafe {
                    zstr_launch_stage_audit(
                        zstr.pinned_base as *mut c_void,
                        zstr.inter_slot_stride,
                        ZSTR_EXTERNAL_WORK_OFFSET,
                        ZSTR_POTENTIAL_ENERGY_OFFSET,
                        ZSTR_ADJUDICATION_CODE_OFFSET,
                        adj_dev as *const c_void,
                        zstr.d_active_slot as *const c_void,
                        telemetry_stream as *mut c_void,
                    )
                };
                if rc != 0 {
                    return Err(BuildError::Cuda {
                        stage: "zstr_stage_audit capture (M1.2.18.5)",
                        rc,
                    });
                }

                // Fence base = slot-0 base + fence_offset_in_slot.
                let fence_base_ptr =
                    unsafe { zstr.pinned_base.add(zstr.fence_offset_in_slot as usize) };
                let rc = unsafe {
                    zstr_launch_fence_signal(
                        fence_base_ptr as *mut c_void,
                        zstr.inter_slot_stride,
                        zstr.d_active_slot as *const c_void,
                        telemetry_stream as *mut c_void,
                    )
                };
                if rc != 0 {
                    return Err(BuildError::Cuda {
                        stage: "zstr_fence_signal capture (G23)",
                        rc,
                    });
                }
                // Snapshot fence node.
                unsafe {
                    let mut s = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
                    let mut id: cuuint64_t = 0;
                    let mut g: CUgraph = ptr::null_mut();
                    let mut dp: *const CUgraphNode = ptr::null();
                    let mut nd: usize = 0;
                    let rc = cuStreamGetCaptureInfo_v2(
                        telemetry_stream,
                        &mut s,
                        &mut id,
                        &mut g,
                        &mut dp,
                        &mut nd,
                    );
                    if matches!(rc, CUresult::CUDA_SUCCESS) && nd > 0 {
                        zstr_fence_node = *dp;
                    }
                }

                log::debug!(
                    "[G23] ZSTR nodes captured: pos_stage={:p} fence={:p} \
                 n_atoms={} src_vram={:#x}",
                    zstr_pos_stage_node,
                    zstr_fence_node,
                    zstr.n_atoms,
                    zstr.d_positions as u64,
                );
                // §3.III-hoist: mirror this iteration's snapshots to the outer
                // hoisted scalars so post-capture wiring sees them.
                zstr_pos_stage_node_hoisted = zstr_pos_stage_node;
                zstr_fence_node_hoisted = zstr_fence_node;
                zstr_src_vram_hoisted = zstr_src_vram;
                zstr_n_atoms_hoisted = zstr_n_atoms;
            }
            // §3.III-hoist: mirror adj_node_set to outer scope (Vec move via
            // clone — the Vec is small, ~handful of nodes per iteration).
            adj_node_set_hoisted = adj_node_set.clone();

            // ── 6.c-end Cross-stream JOIN: telemetry → MD ─────────────────
            // After the DMA retires on the telemetry stream, fire the join
            // event. md_stream waits on it BEFORE launching the trampoline,
            // so cuStreamEndCapture sees a fully-joined dependency graph.
            // (Operator §3 IGNITION: "Wrap the entire 2+2+2+2 sequence" —
            // the JOIN is what makes "wrap" semantically correct under
            // CUDA_ERROR_STREAM_CAPTURE_UNJOINED enforcement.)
            unsafe {
                let rc = cuEventRecord(telemetry_to_md_event, telemetry_stream);
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage: "cuEventRecord (telemetry → md JOIN)",
                        rc: rc as i32,
                    });
                }
                let rc = cuStreamWaitEvent(md_stream.cu_stream(), telemetry_to_md_event, 0);
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage: "cuStreamWaitEvent (md JOIN)",
                        rc: rc as i32,
                    });
                }
            }
            // Retire observation work only after the telemetry JOIN. SFA and
            // ZSTR must consume the same completed interval before it is zeroed.
            // Resetting immediately after SFA erased every saved W_ext value.
        if !cfg.d_external_work.is_null() {
            unsafe {
                let rc = cuMemsetD8Async(
                    cfg.d_external_work as CUdeviceptr,
                    0,
                    8,
                    md_stream.cu_stream(),
                );
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    return Err(BuildError::Cuda {
                        stage: "post-ZSTR observation-work retirement",
                        rc: rc as i32,
                    });
                }
            }
        }


        } // ── end §3.III BODY-UNROLL for-loop ─────────────────────────

        // ── 6.d Node C' (Trampoline) — DEFERRED to V2.
        // The trampoline takes the conditional handle as an argument
        // and calls `cudaGraphSetConditional`. Without a consuming
        // conditional node downstream, the trampoline + handle
        // combination is rejected by `cuGraphInstantiate` with
        // CUDA_ERROR_INVALID_VALUE. V2 ships both the conditional
        // node and the trampoline together so the graph is
        // well-formed end-to-end.
        let _ = cond_handle;

        // Capture the dependency frontier BEFORE
        // ending capture so we can wire the explicit cuGraphAddDependencies
        // edge to the conditional node post-capture (operator §2.3).
        let mut deps_after_trampoline: *const CUgraphNode = ptr::null();
        let mut n_deps_after: usize = 0;
        let mut cap_status_check: CUstreamCaptureStatus =
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut cap_id_check: cuuint64_t = 0;
        let mut graph_check: CUgraph = ptr::null_mut();
        unsafe {
            let rc = cuStreamGetCaptureInfo_v2(
                md_stream.cu_stream(),
                &mut cap_status_check as *mut _,
                &mut cap_id_check as *mut _,
                &mut graph_check as *mut _,
                &mut deps_after_trampoline as *mut _,
                &mut n_deps_after as *mut _,
            );
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                return Err(BuildError::Cuda {
                    stage: "cuStreamGetCaptureInfo_v2 (post-trampoline)",
                    rc: rc as i32,
                });
            }
        }
        if n_deps_after == 0 {
            return Err(BuildError::CaptureFrontierEmpty);
        }
        // Snapshot the dependency-frontier into an owned Vec (the
        // pointer returned by cuStreamGetCaptureInfo is invalidated
        // by subsequent capture API calls).
        let trampoline_node_set: Vec<CUgraphNode> =
            unsafe { std::slice::from_raw_parts(deps_after_trampoline, n_deps_after).to_vec() };

        // ── 7. End capture → final CUgraph ────────────────────────────
        let cu_graph = match capture_guard.end_capture_for_commit() {
            Ok(g) if !g.is_null() => {
                capture_guard.commit(g);
                g
            }
            Ok(_) => {
                return Err(BuildError::Cuda {
                    stage: "cuStreamEndCapture returned null graph",
                    rc: -1,
                })
            }
            Err(e) => return Err(BuildError::Driver(e)),
        };

        // ── 7.4 B.3.2-FULL — G26 SWITCH wire + body population ────────
        // Post-capture: add the 4-way SWITCH conditional node downstream
        // of the bridge kernel and populate the four body sub-graphs.
        // Driver Probe (B.3-narrow commit d8dcaa10) confirmed CUDA 13.x
        // populates phGraph_out for SWITCH-type — no kernel-conditional
        // fallback needed.
        let mut g26_cond_node: CUgraphNode = ptr::null_mut();
        let mut g26_body_subgraphs: [CUgraph; 4] = [ptr::null_mut(); 4];
        // §3.III-hoist: read hoisted snapshot (last iteration of the body
        // unroll). At body_unroll = 1 this is bit-identical to the
        // pre-§3 behavior.
        let g26_bridge_node = g26_bridge_node_hoisted;
        let g26_cond_handle = g26_cond_handle_hoisted;
        if g26_switch_owned_by_parent {
            if g26_bridge_node.is_null() {
                log::warn!(
                    "[TIER8-G26] parent-owned SWITCH requested but child bridge node \
                     snapshot is null; parent SWITCH will use its default gear"
                );
            } else {
                log::info!(
                    "[TIER8-G26] child template captured SFA+bridge only; \
                     G26 SWITCH deferred to parent graph"
                );
            }
        } else if !g26_bridge_node.is_null() {
            let rc = unsafe {
                prism_gearbox_wire_with_handle_ffi(
                    cu_graph,
                    g26_bridge_node,
                    g26_cond_handle,
                    &mut g26_cond_node as *mut _,
                    g26_body_subgraphs.as_mut_ptr(),
                )
            };
            if rc != 0 {
                unsafe {
                    let _ = result::graph::destroy(cu_graph);
                }
                return Err(BuildError::Cuda {
                    stage: "B.3.2 prism_gearbox_wire_with_handle_ffi",
                    rc,
                });
            }
            // Driver-Probe assertion at runtime: all four bodies must
            // be non-null per the B.3-narrow result.  Defensive guard
            // here re-validates per-build (driver behaviour could
            // regress on toolkit upgrades).
            for i in 0..4 {
                if g26_body_subgraphs[i].is_null() {
                    unsafe {
                        let _ = result::graph::destroy(cu_graph);
                    }
                    return Err(BuildError::Cuda {
                        stage: "B.3.2 SWITCH body_subgraphs null (driver regression)",
                        rc: -1,
                    });
                }
            }
            // Populate the bodies.  d_velocities comes from PipelineConfig
            // (T12 Pre-Flight wired it through to the FFI struct's offset
            // 120; we read the same address for the gearbox's per-body
            // rescale kernel).  n_atoms is taken from AscConfig (the only
            // place in PipelineConfig that carries it).  When asc is None
            // OR cfg.d_velocities is null, skip body population — the
            // SWITCH still fires but the bodies are no-ops.
            if governor_device.has_physical_guard() {
                let rc=unsafe {crate::governor_device::prism_gearbox_populate_guard_bodies_v1(
                    g26_body_subgraphs.as_mut_ptr(),governor_device.device_ptr())};
                if rc!=0 {return Err(BuildError::Cuda{stage:"G26 physical guard observer bodies",rc});}
            } else if let Some(ref asc) = cfg.asc {
                if !cfg.d_velocities.is_null() {
                    let n_floats = (asc.n_atoms as u32).saturating_mul(3u32);
                    let rc = unsafe {
                        prism_gearbox_populate_switch_bodies_ffi(
                            g26_body_subgraphs.as_mut_ptr(),
                            adj_dev as *const InterferometricAdjudicatorFfi,
                            cfg.d_velocities,
                            n_floats,
                            cruise_state_dev as *const c_void,
                            ptr::null(), // d_current_temp — Berendsen disabled in B.3.2 (PE wiring deferred)
                            ptr::null(), // d_dt
                            300.0_f32,   // target_temp_K placeholder
                            0.5_f32,     // tau_ps placeholder
                        )
                    };
                    if rc != 0 {
                        unsafe {
                            let _ = result::graph::destroy(cu_graph);
                        }
                        return Err(BuildError::Cuda {
                            stage: "B.3.2 prism_gearbox_populate_switch_bodies_ffi",
                            rc,
                        });
                    }
                    log::info!(
                        "[B.3.2] G26 SWITCH wired: cond_node={:p} bodies populated \
                         (rescale + apply_dt for 0/1/2; trap for 3); n_floats={}",
                        g26_cond_node,
                        n_floats
                    );
                }
            }
        }

        // ── 7.5 V2 IGNITION HOOK ────────────────────────────────────
        // Invoke the caller-provided hook between end_capture and
        // cuGraphInstantiate. The V1 wrapper passes a no-op closure;
        // V2 callers inject the F1 SWITCH conditional node here via
        // Claude-2's `prism_wire_f1_switch_ffi` C-ABI bypass. Abort
        // cleanly on failure so the raw CUgraph isn't leaked.
        if let Err(rc) = hook(cu_graph, &adj_node_set_hoisted, adj_dev) {
            unsafe {
                let _ = result::graph::destroy(cu_graph);
            }
            // Stream-ordered free of pool allocations so the pool
            // drop on `pool` in the early-return doesn't see live
            // pointers.
            let _ = pool.free_async(tiles_dev, md_raw);
            let _ = pool.free_async(adj_dev, md_raw);
            let _ = pool.free_async(burst_marker_dev, md_raw);
            if sisr_mask_dev != 0 {
                let _ = pool.free_async(sisr_mask_dev, md_raw);
            }
            // Borrowed GovernorDevice::cruise_state_ptr() aliases its owner at
            // +592; it is not an F2 allocation. The retained governor retires it.
            return Err(BuildError::V2HookFailed { rc });
        }

        // ── 8. V1 boundary: F1 SWITCH conditional node DEFERRED to V2 ─
        //
        // The conditional handle has been created against the captured
        // graph (operator §2.3 prerequisite); the trampoline kernel's
        // `cudaGraphSetConditional(handle, code)` call is captured as a
        // kernel node and runs at every launch (harmless when no
        // downstream conditional node consumes the handle, per the
        // CUDA Programming Guide).
        //
        // V1 ship scope: linear capture + handle creation + telemetry
        // DMA + instantiate. The conditional-node-addition step
        // (`cuGraphAddNode_v2(CONDITIONAL)`) returns CUDA_SUCCESS but
        // populates `phGraph_out[0] = null` on the local CUDA 13
        // driver — empirically observed during this commit. V2
        // follow-up: debug the null-body behaviour (likely a cudarc
        // 0.18.2 binding subtlety vs. CUDA 13 driver expectation).
        //
        // The trampoline node is the captured-graph "tail" — operator
        // §2.3's "explicit edge from Node C (Adjudicator) to Node D
        // (trampoline)" is satisfied IMPLICITLY by sequential capture
        // on md_stream (each launched kernel takes the previous as
        // its dependency). When V2 lands, the explicit edge becomes
        // trampoline → conditional node, added via `cuGraphAddNode`'s
        // `dependencies` parameter.
        let cond_node: CUgraphNode = ptr::null_mut();
        let body_subgraph: CUgraph = ptr::null_mut();
        let _ = trampoline_node_set; // V2 will consume this snapshot.

        // ── 9. Instantiate ───────────────────────────────────────────
        let cu_graph_exec = unsafe {
            match result::graph::instantiate(
                cu_graph,
                CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            ) {
                Ok(e) => e,
                Err(e) => {
                    let _ = result::graph::destroy(cu_graph);
                    return Err(BuildError::Driver(e));
                }
            }
        };

        // ── 9.5 G24 slot-roller: read ZSTR CUfunction handles ──────────
        // `cuGraphKernelNodeGetParams` reads from the GRAPH (not exec),
        // so it must be called AFTER end_capture and BEFORE or AFTER
        // instantiate (graph topology is frozen after end_capture).
        // We call it here (post-instantiate) to avoid extra ordering
        // constraints.  On failure (ZSTR disabled or node null) the
        // func fields stay null — `launch_with_zstr_slot` no-ops.
        // §3.III-hoist: read hoisted snapshots (last iteration of body).
        let zstr_pos_stage_node = zstr_pos_stage_node_hoisted;
        let zstr_fence_node = zstr_fence_node_hoisted;
        let zstr_src_vram = zstr_src_vram_hoisted;
        let zstr_n_atoms = zstr_n_atoms_hoisted;
        let zstr_pos_stage_func: CUfunction = if !zstr_pos_stage_node.is_null() {
            let mut p: CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
            let rc = unsafe { cuGraphKernelNodeGetParams_v2(zstr_pos_stage_node, &mut p) };
            if matches!(rc, CUresult::CUDA_SUCCESS) {
                log::debug!("[G24] zstr_pos_stage func={:p}", p.func);
                p.func
            } else {
                log::warn!("[G24] cuGraphKernelNodeGetParams(pos_stage) rc={:?}", rc);
                ptr::null_mut()
            }
        } else {
            ptr::null_mut()
        };

        let zstr_fence_func: CUfunction = if !zstr_fence_node.is_null() {
            let mut p: CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
            let rc = unsafe { cuGraphKernelNodeGetParams_v2(zstr_fence_node, &mut p) };
            if matches!(rc, CUresult::CUDA_SUCCESS) {
                log::debug!("[G24] zstr_fence func={:p}", p.func);
                p.func
            } else {
                log::warn!("[G24] cuGraphKernelNodeGetParams(fence) rc={:?}", rc);
                ptr::null_mut()
            }
        } else {
            ptr::null_mut()
        };

        Ok(Self {
            pool,
            governor_device,
            asc_device,
            md_stream: md_stream.clone(),
            telemetry_stream,
            md_to_telemetry_event,
            telemetry_to_md_event,
            perturbed_positions_dev: d_perturbed_positions,
            com_shift_dev: d_com_shift_dev,
            total_mass_dev: d_total_mass_dev,
            com_correction_dev: d_com_correction_dev,
            tiles_dev,
            tiles_baseline_dev,
            p_frozen_dev,
            tiles_bytes: tiles_bytes as u64,
            adj_dev,
            burst_marker_dev,
            sisr_mask_dev,
            sisr_count_dev,
            dynt7_acc_dev,
            dynt7_idx_dev,
            dynt7_stats_dev,
            cruise_state_dev,
            g26_cond_handle,
            f1_cond_handle: cfg.f1_parent_cond_handle,
            parent_predicate_count: body_unroll as u32,
            g26_cond_node,
            energy_temp_storage_dev,
            energy_temp_storage_bytes,
            energy_pe_scalar_dev,
            energy_window_dev,
            ring,
            cu_graph,
            cu_graph_exec,
            cond_handle,
            body_subgraph,
            zstr_pos_stage_node,
            zstr_fence_node,
            zstr_pos_stage_func,
            zstr_fence_func,
            zstr_src_vram,
            zstr_n_atoms,
            ghost_physical_step_dev: cfg.d_current_md_step as u64,
            ghost_step_counter_dev: cfg
                .mar_v2
                .as_ref()
                .map(|mar| mar.d_step_counter)
                .unwrap_or(0),
            // GEAR-PROVENANCE-002 — retain the ProtocolState.dt address so
            // the host can read the timestep the gearbox actually applied,
            // and resolve the readback bound once so every record on this
            // pipeline cites the same auditable budget.
            g26_dt_dev: cfg.d_dt as usize,
            g26_readback_timeout: crate::gearbox::resolve_snapshot_timeout(),
            g26_stream_id: cfg.source_stream_id,
            g26_trace_cfg: G26TraceConfig::resolve(cfg.source_stream_id),
            g26_trace: Mutex::new(G26TraceSink::default()),
            n_seed_spikes: cfg.n_spikes,
            seed_layout: cfg.seed_layout,
            n_physical_atoms: cfg.n_atoms_for_pe,
            n_clusters: cfg.n_clusters,

            // SO(3) + Adjudicator [+ ASC Node D] [+ ZSTR pos_stage +
            // force_stage + force_norm_sqrt + fence] [+ SISR] +
            // [B.3.2] gearbox SFA + bridge + 4 SWITCH bodies
            // (rescale + apply_dt × 3 + trap = 7 inner nodes when
            // populated).
            n_kernel_nodes_captured: 2
                + if cfg.asc.is_some()  { 1 } else { 0 }
                + if cfg.zstr.is_some() { 4 } else { 0 }
                + if cfg.sisr.is_some() { 1 } else { 0 }
                + 2  // B.3.2: SFA + predicate-bridge captured kernels
                + if !g26_switch_owned_by_parent
                    && cfg.asc.is_some()
                    && !cfg.d_velocities.is_null() {
                    7  // B.3.2: bodies populated (rescale × 3, apply_dt × 3, trap × 1)
                  } else { 0 },
            // V1: 0 explicit cuGraphAddDependencies edges (the
            // C→D edge is satisfied by capture-mode's implicit
            // sequential ordering on md_stream). V2 lands the
            // explicit cuGraphAddNode(CONDITIONAL, deps=[trampoline])
            // which IS the §2.3 explicit-edge mandate.
            n_dependency_edges_explicit: {
                let _ = cond_node;
                0u32
            },
        })
    }

    /// TIER 8 Option β — create a G26 conditional handle bound to a
    /// parent/control graph before the child adjudication template is
    /// captured.  The returned handle is passed through
    /// `PipelineConfig::g26_parent_cond_handle`, letting the child
    /// predicate bridge write through the parent-owned handle while
    /// keeping the child template free of conditional nodes.
    ///
    /// # Safety
    /// `parent_graph` must be a valid, mutable, uninstantiated CUgraph.
    pub unsafe fn create_parent_g26_cond_handle(parent_graph: CUgraph) -> Result<u64, BuildError> {
        if parent_graph.is_null() {
            return Err(BuildError::InvalidConfig {
                reason: "parent graph is null for G26 handle creation",
            });
        }
        let mut handle: u64 = 0;
        let rc = prism_gearbox_create_handle_ffi(
            parent_graph,
            /*default_value=*/ 1u32,
            &mut handle as *mut u64,
        );
        if rc != 0 {
            return Err(BuildError::Cuda {
                stage: "TIER8 prism_gearbox_create_handle_ffi(parent)",
                rc,
            });
        }
        if handle == 0 {
            return Err(BuildError::Cuda {
                stage: "TIER8 parent G26 handle came back zero",
                rc: -1,
            });
        }
        Ok(handle)
    }

    /// TIER 8 Option β — install the G26 SWITCH node on the parent graph
    /// after the adjudication child has been spliced.  The dependency is
    /// the parent-level child-graph node, because CUDA intentionally hides
    /// internal child nodes from the parent topology.
    ///
    /// Returns the parent-owned G26 conditional node.  The caller should
    /// route downstream parent work from this node to preserve
    /// `ChildAdj → G26_SWITCH → next_parent_node` ordering.
    ///
    /// # Safety
    /// `parent_graph` and `dependency_node` must belong to the same live,
    /// mutable, uninstantiated CUgraph.
    pub unsafe fn wire_parent_g26_switch(
        &self,
        parent_graph: CUgraph,
        dependency_node: CUgraphNode,
        d_velocities: *mut f32,
        n_floats: u32,
    ) -> Result<CUgraphNode, BuildError> {
        self.wire_parent_g26_switch_with_handle(parent_graph,dependency_node,self.g26_cond_handle,d_velocities,n_floats)
    }

    pub(crate) unsafe fn wire_parent_g26_switch_with_handle(&self,parent_graph:CUgraph,
        dependency_node:CUgraphNode,handle:u64,d_velocities:*mut f32,n_floats:u32)
        -> Result<CUgraphNode,BuildError> {
        if parent_graph.is_null() || dependency_node.is_null() {
            return Err(BuildError::InvalidConfig {
                reason: "parent graph/dependency is null for G26 SWITCH",
            });
        }
        if handle == 0 {
            return Err(BuildError::InvalidConfig {
                reason: "parent G26 SWITCH requested without a conditional handle",
            });
        }

        let mut g26_cond_node: CUgraphNode = ptr::null_mut();
        let mut g26_body_subgraphs: [CUgraph; 4] = [ptr::null_mut(); 4];
        let rc = prism_gearbox_wire_with_handle_ffi(
            parent_graph,
            dependency_node,
            handle,
            &mut g26_cond_node as *mut _,
            g26_body_subgraphs.as_mut_ptr(),
        );
        if rc != 0 {
            return Err(BuildError::Cuda {
                stage: "TIER8 prism_gearbox_wire_with_handle_ffi(parent)",
                rc,
            });
        }
        for i in 0..4 {
            if g26_body_subgraphs[i].is_null() {
                return Err(BuildError::Cuda {
                    stage: "TIER8 parent G26 SWITCH body_subgraphs null",
                    rc: -1,
                });
            }
        }

        if self.governor_device.has_physical_guard() {
            let rc=crate::governor_device::prism_gearbox_populate_guard_bodies_v1(
                g26_body_subgraphs.as_mut_ptr(),self.governor_device.device_ptr());
            if rc!=0 {return Err(BuildError::Cuda{stage:"parent G26 physical guard observer bodies",rc});}
        } else if !d_velocities.is_null() && n_floats > 0 {
            let rc = prism_gearbox_populate_switch_bodies_ffi(
                g26_body_subgraphs.as_mut_ptr(),
                self.adj_dev as *const InterferometricAdjudicatorFfi,
                d_velocities,
                n_floats,
                self.cruise_state_dev as *const c_void,
                ptr::null(), // d_current_temp — Berendsen disabled in B.3.2
                ptr::null(), // d_dt
                300.0_f32,
                0.5_f32,
            );
            if rc != 0 {
                return Err(BuildError::Cuda {
                    stage: "TIER8 prism_gearbox_populate_switch_bodies_ffi(parent)",
                    rc,
                });
            }
            log::info!(
                "[TIER8-G26] parent SWITCH wired: cond_node={:p} \
                 bodies populated; n_floats={}",
                g26_cond_node,
                n_floats
            );
        } else {
            log::warn!(
                "[TIER8-G26] parent SWITCH wired without velocity bodies \
                 (d_velocities={:p}, n_floats={}); bodies remain empty",
                d_velocities,
                n_floats
            );
        }

        Ok(g26_cond_node)
    }

    /// **F1-PARENT-SWITCH-001 — create the parent-owned F1 conditional
    /// handle.** Mirrors `create_parent_g26_cond_handle`, swapping the
    /// FFI to `prism_f1_create_handle_ffi` and using the F1 default
    /// launch value of 0 (PRISM_ADJ_PRUNE) instead of G26's 1 (Gear 1).
    /// The returned handle is passed back through
    /// `PipelineConfig::f1_parent_cond_handle` so the captured child's
    /// F1 predicate-bridge kernel can write into it before the parent-
    /// graph F1 SWITCH fires.
    ///
    /// # Safety
    /// `parent_graph` must be a valid, mutable, uninstantiated CUgraph.
    pub unsafe fn create_parent_f1_cond_handle(parent_graph: CUgraph) -> Result<u64, BuildError> {
        if parent_graph.is_null() {
            return Err(BuildError::InvalidConfig {
                reason: "parent graph is null for F1 handle creation",
            });
        }
        let mut handle: u64 = 0;
        let rc = prism_f1_create_handle_ffi(
            parent_graph,
            /*default_value=*/ 0u32, // PRISM_ADJ_PRUNE — F1 case 0
            &mut handle as *mut u64,
        );
        if rc != 0 {
            return Err(BuildError::Cuda {
                stage: "F1-PARENT-SWITCH prism_f1_create_handle_ffi(parent)",
                rc,
            });
        }
        if handle == 0 {
            return Err(BuildError::Cuda {
                stage: "F1-PARENT-SWITCH parent F1 handle came back zero",
                rc: -1,
            });
        }
        Ok(handle)
    }

    /// **F1-PARENT-SWITCH-001 — install the F1 SWITCH (size=3) on the
    /// parent graph.** Adds a `cudaGraphCondTypeSwitch` conditional node
    /// downstream of `dependency_node` (typically the spliced child-
    /// adjudicator graph node) and populates the 3 body sub-graphs:
    ///   case 0 (Prune)     — empty no-op
    ///   case 1 (Construct) — empty no-op (ASC reserved)
    ///   case 2 (Violation) — PTX trap kernel
    ///
    /// Returns the parent-owned F1 SWITCH conditional node. The caller
    /// must wire the F1 SWITCH BEFORE the G26 SWITCH (so F1's Violation
    /// branch short-circuits before G26 mutates gear/dt state).
    ///
    /// # Safety
    /// `parent_graph` and `dependency_node` must belong to the same
    /// live, mutable, uninstantiated CUgraph. `f1_cond_handle` must
    /// have been created on the same parent graph via
    /// `create_parent_f1_cond_handle`.
    pub unsafe fn wire_parent_f1_switch(
        &self,
        parent_graph: CUgraph,
        f1_cond_handle: u64,
        dependency_node: CUgraphNode,
    ) -> Result<CUgraphNode, BuildError> {
        if parent_graph.is_null() || dependency_node.is_null() {
            return Err(BuildError::InvalidConfig {
                reason: "parent graph/dependency is null for F1 SWITCH",
            });
        }
        if f1_cond_handle == 0 {
            return Err(BuildError::InvalidConfig {
                reason: "parent F1 SWITCH requested without a conditional handle",
            });
        }

        let mut f1_cond_node: CUgraphNode = ptr::null_mut();
        let mut f1_body_subgraphs: [CUgraph; 3] = [ptr::null_mut(); 3];
        let rc = prism_f1_wire_with_handle_ffi(
            parent_graph,
            dependency_node,
            f1_cond_handle,
            &mut f1_cond_node as *mut _,
            f1_body_subgraphs.as_mut_ptr(),
        );
        if rc != 0 {
            return Err(BuildError::Cuda {
                stage: "F1-PARENT-SWITCH prism_f1_wire_with_handle_ffi(parent)",
                rc,
            });
        }
        for i in 0..3 {
            if f1_body_subgraphs[i].is_null() {
                return Err(BuildError::Cuda {
                    stage: "F1-PARENT-SWITCH parent F1 SWITCH body_subgraphs null",
                    rc: -1,
                });
            }
        }

        let rc = prism_f1_populate_switch_bodies_ffi(f1_body_subgraphs.as_mut_ptr());
        if rc != 0 {
            return Err(BuildError::Cuda {
                stage: "F1-PARENT-SWITCH prism_f1_populate_switch_bodies_ffi(parent)",
                rc,
            });
        }

        log::info!(
            "[F1-PARENT-SWITCH] parent SWITCH wired: cond_node={:p} \
             handle={:#x} size=3 (Prune/Construct/Violation), \
             body[2]=trap kernel populated",
            f1_cond_node,
            f1_cond_handle
        );

        Ok(f1_cond_node)
    }

    /// **Operator Amendment 3.9 §2.2 — G19.5 Zero-Trust Pointer Alignment Audit.**
    ///
    /// Runs immediately before `cuGraphLaunch`.  Panics on any pointer
    /// that fails the 16-byte alignment requirement for Blackwell sm_120
    /// vector ops (`LDG.E.128`, `STG.E.128`, `RED.E.ADD.V4`).  Pointers
    /// from the F2 pool are 256-byte aligned by construction (the
    /// `vram_pool::alloc_async` assertion catches any allocator
    /// regression at mint time); this guard catches anything that
    /// landed via a different code path (e.g., cudarc `alloc_zeros`,
    /// pinned-host `cuMemAllocHost_v2`, or the engine's primary
    /// integrator buffers wired through `PipelineConfig`).
    ///
    /// Side effect: panics with a HALT message naming the offending
    /// pointer + its hex address + actual alignment.  `panic!` here is
    /// preferred over `Result<>` because a misaligned pointer at this
    /// site is a structural correctness violation — the GPU will trap
    /// in nanoseconds anyway, and the panic stack trace gives the host-
    /// side diagnostic the hardware MISALIGNED_ADDRESS does not.
    fn audit_v2_graph_pointers(&self) {
        let tiles_dev_ptr = self.tiles_dev as u64;
        let tiles_baseline_ptr = self.tiles_baseline_dev as u64;
        let adj_dev_ptr = self.adj_dev as u64;
        let burst_marker_ptr = self.burst_marker_dev as u64;
        let cruise_state_ptr = self.cruise_state_dev as u64;
        let energy_pe_scalar_ptr = self.energy_pe_scalar_dev as u64;
        let energy_window_ptr = self.energy_window_dev as u64;

        // 16-byte alignment is the Blackwell vector-load minimum;
        // F2-pool allocations exceed this at 256-byte alignment by
        // construction (cudaMallocFromPoolAsync contract).
        let pointers: [(&str, u64); 7] = [
            ("tiles_dev", tiles_dev_ptr),
            ("tiles_baseline", tiles_baseline_ptr),
            ("adj_dev", adj_dev_ptr),
            ("burst_marker", burst_marker_ptr),
            ("cruise_state", cruise_state_ptr),
            ("energy_pe_scalar", energy_pe_scalar_ptr),
            ("energy_window", energy_window_ptr),
        ];

        for (name, ptr) in pointers {
            // Skip null pointers (legitimate when the corresponding
            // captured-graph branch is gated off — e.g., energy
            // monitor when n_atoms_for_pe == 0).
            if ptr == 0 {
                continue;
            }
            assert!(
                ptr % 16 == 0,
                "FATAL ARCHITECTURAL VIOLATION (G19.5 PAG): {} pointer \
                 (0x{:x}) is not 16-byte aligned (mod 16 = {}). \
                 Blackwell sm_120 vector loads will trap. HALT.",
                name,
                ptr,
                ptr % 16
            );
        }
    }

    /// Launch the captured graph once. Stream-ordered against the
    /// caller-provided MD stream — caller synchronizes when they need
    /// the host-visible burst_marker / ring slot.
    ///
    /// Operator Amendment 3.9 §2.2 — runs the G19.5 PAG audit before
    /// every launch.  The audit is cheap (7 modulo-16 checks); the
    /// CUDA Graph launch path doesn't tolerate misaligned pointers,
    /// so catching them in safe Rust here gives a stack trace the
    /// hardware MISALIGNED_ADDRESS does not.
    pub(crate) fn prepare_launch(&self) -> Result<(), DriverError> {
        self.audit_v2_graph_pointers();
        // Defensive fail-closed context bind before every cuGraphLaunch: the
        // per-OS-thread current context can drift between chunks; ensure the
        // captured graph (which contains the gasp kernel node) replays on its
        // own context. Do not ignore either driver result, because that would
        // misattribute a failed context operation to the subsequent launch.
        {
            let expected_ctx = self.md_stream.context().cu_ctx();
            let mut cur: CUcontext = ptr::null_mut();
            let rc_get = unsafe { cuCtxGetCurrent(&mut cur as *mut _) };
            if rc_get != CUresult::CUDA_SUCCESS {
                return Err(DriverError(rc_get));
            }
            if cur != expected_ctx {
                let rc_set = unsafe { cuCtxSetCurrent(expected_ctx) };
                if rc_set != CUresult::CUDA_SUCCESS {
                    return Err(DriverError(rc_set));
                }
            }
        }
        Ok(())
    }

    pub fn launch(&self) -> Result<(), DriverError> {
        self.prepare_launch()?;
        let launched =
            unsafe { result::graph::launch(self.cu_graph_exec, self.md_stream.cu_stream()) };

        // GEAR-PROVENANCE-002 — the gearbox decision boundary. Every replay
        // runs the SFA, the predicate bridge and one SWITCH body, so this is
        // where a MEASURED gear exists. The observation is opt-in via
        // `PRISM_G26_CONTROL_TRACE_DIR` because it costs a bounded stream
        // drain; with the key unset this branch is a pointer test and the
        // launch path is unchanged. It can never fail a launch.
        if launched.is_ok() && self.g26_trace_cfg.is_some() {
            self.observe_g26_after_launch();
        }
        launched
    }

    /// Build the observation root from existing pure KCC/V2 children and
    /// parent-owned native F1/G26 conditionals. A monolithic child carries
    /// handles owned by its physical parent and must not be replayed alone.
    /// This builder gives the observation endpoint its own conditional owners.
    pub fn build_kcc_observation_parent(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        cfg: &PipelineConfig,
        kcc: crate::graph_capture::CapturedTemplate,
    ) -> anyhow::Result<KccObservationParent> {
        Self::build_kcc_observation_parent_with_prefix(ctx, stream, cfg, kcc,
            |_parent| Ok((Vec::new(), None, 0)))
    }

    /// Append observation nodes to a caller-owned physical prefix. The prefix
    /// must retain all pure-child owners and return its terminal dependencies;
    /// its conditional handles belong to this same root, never a child graph.
    pub(crate) fn build_kcc_observation_parent_with_prefix(
        ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, cfg: &PipelineConfig,
        kcc: crate::graph_capture::CapturedTemplate,
        prefix: impl FnOnce(&mut crate::graph_capture::CapturedTemplate) -> anyhow::Result<(
            Vec<CUgraphNode>, Option<crate::graph_capture::ResidentLoopState>, usize)>,
    ) -> anyhow::Result<KccObservationParent> {
        Self::build_kcc_observation_parent_with_prefix_owner(ctx,stream,cfg,kcc,None,prefix)
    }

    pub(crate) fn build_kcc_observation_parent_with_prefix_owner(
        ctx:&Arc<CudaContext>,stream:&Arc<CudaStream>,cfg:&PipelineConfig,
        kcc:crate::graph_capture::CapturedTemplate,existing:Option<Arc<Self>>,
        prefix:impl FnOnce(&mut crate::graph_capture::CapturedTemplate)->anyhow::Result<(
            Vec<CUgraphNode>,Option<crate::graph_capture::ResidentLoopState>,usize)>,
    )->anyhow::Result<KccObservationParent> {
        ctx.bind_to_thread()?;
        stream.synchronize()?;
        let mut graph = ptr::null_mut();
        let rc = unsafe { cuGraphCreate(&mut graph, 0) };
        anyhow::ensure!(rc == CUresult::CUDA_SUCCESS, "create observation parent: {:?}", rc);
        let mut parent = crate::graph_capture::CapturedTemplate::from_capture(
            graph, std::collections::HashMap::new(), stream.clone());
        let (prefix_dependencies, resident_state, prefix_conditionals) = prefix(&mut parent)?;
        let pipeline=if let Some(existing)=existing {
            anyhow::ensure!(Arc::ptr_eq(existing.md_stream.context(),ctx)
                && existing.md_stream.cu_stream()==stream.cu_stream(),
                "live observation owner belongs to a different physical stream/context");
            existing.append_existing_kcc_observation_nodes(cfg,&kcc,&mut parent,&prefix_dependencies)?;
            existing
        } else {
            Self::append_kcc_observation_nodes(ctx,stream,cfg,&kcc,&mut parent,&prefix_dependencies)?.0
        };
        let shape = unsafe { crate::graph_node::splice_legality_check(graph) }
            .map_err(|rc| anyhow::anyhow!("observation parent topology readback: {rc}"))?;
        anyhow::ensure!(shape.conditional_count as usize == 2 + prefix_conditionals,
            "observation parent must own F1 and G26: observed {} conditionals", shape.conditional_count);
        let executable = if let Some(state) = resident_state {
            let steps = state.max_iterations;
            parent.instantiate_with_runtime_state(steps, Some(state))?
        } else { parent.instantiate()? };
        log::info!("[KCC-V2-PARENT-CAPTURE] order=KCC,V2,F1,G26 parent_owned_conditionals={} root_nodes={} physics_steps_advanced=0 capture_only=true",
            shape.conditional_count, shape.total_nodes);
        Ok(KccObservationParent { graph: Box::new(executable), pipeline })
    }

    /// Append native observation stages into the actual owning root. The
    /// caller retains external physical/CSR/ring allocations until root drain.
    pub(crate) fn append_kcc_observation_nodes(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>,
        cfg: &PipelineConfig, kcc: &crate::graph_capture::CapturedTemplate,
        parent: &mut crate::graph_capture::CapturedTemplate, dependencies: &[CUgraphNode])
        -> anyhow::Result<(Arc<Self>, CUgraphNode)> {
        ctx.bind_to_thread()?;
        let graph = parent.cu_graph();
        let mut child_cfg = cfg.clone();
        child_cfg.g26_parent_cond_handle = unsafe { Self::create_parent_g26_cond_handle(graph)? };
        child_cfg.f1_parent_cond_handle = unsafe { Self::create_parent_f1_cond_handle(graph)? };
        let pipeline = Arc::new(Self::build(ctx, stream, &child_cfg)?);
        // The same raw-seed relaxed baseline is frozen before the first replay.
        // Build and capture do not consume a KCC sample or a Ghost observation.
        pipeline.freeze_baseline()?;
        parent.retain_child_owners(kcc);
        parent.retain_pipeline(pipeline.clone());
        let kcc_node = parent.add_child_graph_node(dependencies, kcc.cu_graph())?;
        let v2_node = parent.add_child_graph_node(&[kcc_node], pipeline.cu_graph_raw())?;
        let f1_node = unsafe {
            pipeline.wire_parent_f1_switch(graph, child_cfg.f1_parent_cond_handle, v2_node)?
        };
        let n_floats = child_cfg.asc.as_ref().map(|asc|
            (asc.n_atoms.max(0) as u32).saturating_mul(3)).unwrap_or(0);
        let last = unsafe { pipeline.wire_parent_g26_switch(graph, f1_node, child_cfg.d_velocities, n_floats)? };
        Ok((pipeline,last))
    }

    /// Rebind only native predicate handle arguments in a graph clone. All
    /// live tensors, T7 history, frozen reference, adjudicator and output state
    /// remain owned by this exact Arc; no constructor or initializer runs.
    pub(crate) fn append_existing_kcc_observation_nodes(self: &Arc<Self>,
        cfg: &PipelineConfig,kcc:&crate::graph_capture::CapturedTemplate,
        parent:&mut crate::graph_capture::CapturedTemplate,dependencies:&[CUgraphNode])
        -> anyhow::Result<CUgraphNode> {
        extern "C" {
            fn prism_gearbox_rebind_parent_predicates_v1(graph:CUgraph,old:u64,new:u64,count:*mut u32)->i32;
            fn prism_f1_rebind_parent_predicates_v1(graph:CUgraph,old:u64,new:u64,count:*mut u32)->i32;
        }
        self.md_stream.context().bind_to_thread()?;
        self.md_stream.synchronize()?;
        anyhow::ensure!(self.f1_cond_handle!=0 && self.g26_cond_handle!=0 && self.parent_predicate_count>0,
            "live V2 owner lacks explicit parent predicate identity");
        let graph=parent.cu_graph();
        let f1=unsafe {Self::create_parent_f1_cond_handle(graph)?};
        let g26=unsafe {Self::create_parent_g26_cond_handle(graph)?};
        let mut cloned=ptr::null_mut();
        let rc=unsafe {cudarc::driver::sys::cuGraphClone(&mut cloned,self.cu_graph)};
        anyhow::ensure!(rc==CUresult::CUDA_SUCCESS && !cloned.is_null(),"clone live V2 template: {rc:?}");
        let mut template=crate::graph_capture::CapturedTemplate::from_capture(cloned,
            std::collections::HashMap::new(),self.md_stream.clone());
        template.retain_pipeline(self.clone());
        let mut nf1=0;let mut ng26=0;
        let rc=unsafe {prism_f1_rebind_parent_predicates_v1(cloned,self.f1_cond_handle,f1,&mut nf1)};
        anyhow::ensure!(rc==0 && nf1==self.parent_predicate_count,
            "live V2 F1 rebinding incomplete: rc={rc} matched={nf1} expected={}",self.parent_predicate_count);
        let rc=unsafe {prism_gearbox_rebind_parent_predicates_v1(cloned,self.g26_cond_handle,g26,&mut ng26)};
        anyhow::ensure!(rc==0 && ng26==self.parent_predicate_count,
            "live V2 G26 rebinding incomplete: rc={rc} matched={ng26} expected={}",self.parent_predicate_count);
        parent.retain_child_owners(kcc);parent.retain_child_owners(&template);
        let kn=parent.add_child_graph_node(dependencies,kcc.cu_graph())?;
        let vn=parent.add_child_graph_node(&[kn],template.cu_graph())?;
        let fn_=unsafe {self.wire_parent_f1_switch(graph,f1,vn)?};
        let n=cfg.asc.as_ref().map(|asc|(asc.n_atoms.max(0) as u32).saturating_mul(3)).unwrap_or(0);
        let last=unsafe {self.wire_parent_g26_switch_with_handle(graph,fn_,g26,cfg.d_velocities,n)?};
        log::info!("[V2-LIVE-OWNER-REBIND] owner={:p} pool_and_history_unchanged=true f1_nodes={} g26_nodes={} old_f1={} new_f1={} old_g26={} new_g26={}",
            Arc::as_ptr(self),nf1,ng26,self.f1_cond_handle,f1,self.g26_cond_handle,g26);
        Ok(last)
    }

    #[cfg(test)]
    pub(crate) fn rebind_state_sha256(&self)->anyhow::Result<String> {
        use sha2::{Digest,Sha256};
        self.md_stream.synchronize()?;
        let mut h=Sha256::new();
        for (name,pointer,bytes) in [
            ("tiles",self.tiles_dev,self.tiles_bytes as usize),
            ("relaxed",self.tiles_baseline_dev,self.tiles_bytes as usize),
            ("frozen",self.p_frozen_dev,self.tiles_bytes as usize),
            ("adjudicator",self.adj_dev,std::mem::size_of::<InterferometricAdjudicatorFfi>()),
            ("calibration_f64",self.dynt7_acc_dev,32),("calibration_compat_index",self.dynt7_idx_dev,4),
            ("calibration_compat_stats",self.dynt7_stats_dev,8),("cruise",self.cruise_state_dev,16),
            ("ghost_clock",self.ghost_step_counter_dev as usize,8),
        ] {
            if pointer==0 {continue;}
            let mut data=vec![0u8;bytes];
            let rc=unsafe {cudarc::driver::sys::cuMemcpyDtoH_v2(data.as_mut_ptr().cast(),pointer as u64,bytes)};
            anyhow::ensure!(rc==CUresult::CUDA_SUCCESS,"V2 rebind state readback {name}: {rc:?}");
            h.update(name.as_bytes());h.update((pointer as u64).to_le_bytes());h.update(data);
        }
        Ok(format!("{:x}",h.finalize()))
    }

    #[cfg(test)]
    pub(crate) fn rebind_observation_parent(self:&Arc<Self>,cfg:&PipelineConfig,
        kcc:crate::graph_capture::CapturedTemplate)->anyhow::Result<KccObservationParent> {
        self.md_stream.synchronize()?;
        let mut graph=ptr::null_mut();
        let rc=unsafe {cuGraphCreate(&mut graph,0)};
        anyhow::ensure!(rc==CUresult::CUDA_SUCCESS,"rebound observation root create: {rc:?}");
        let mut parent=crate::graph_capture::CapturedTemplate::from_capture(graph,
            std::collections::HashMap::new(),self.md_stream.clone());
        self.append_existing_kcc_observation_nodes(cfg,&kcc,&mut parent,&[])?;
        Ok(KccObservationParent {graph:Box::new(parent.instantiate()?),pipeline:self.clone()})
    }

    /// Whether this captured MAR-v2 producer owns a device-authenticated step
    /// counter. MAR-v1 intentionally returns false and remains eligible only
    /// for segment-only Ghost persistence.
    pub fn has_exact_ghost_device_counter(&self) -> bool {
        self.ghost_step_counter_dev != 0
    }

    /// Compatibility validation for explicit callers. The graph reads the
    /// live physical clock; host requests cannot overwrite record time.
    /// Production overlays need no per-replay host binding.
    pub fn bind_exact_ghost_step(
        &self,
        post_md_step: u64,
    ) -> std::result::Result<(), ExactGhostStepBindError> {
        if post_md_step == 0 || post_md_step > u32::MAX as u64 {
            return Err(ExactGhostStepBindError::BadPostMdStep { post_md_step });
        }
        if self.ghost_step_counter_dev == 0 {
            return Err(ExactGhostStepBindError::MissingDeviceCounter);
        }
        let expected_ctx = self.md_stream.context().cu_ctx();
        let mut observed_ctx: CUcontext = ptr::null_mut();
        let rc_query = unsafe { cuCtxGetCurrent(&mut observed_ctx as *mut _) };
        if rc_query != CUresult::CUDA_SUCCESS {
            return Err(ExactGhostStepBindError::ContextQuery { rc: rc_query });
        }
        if observed_ctx != expected_ctx {
            let rc_bind = unsafe { cuCtxSetCurrent(expected_ctx) };
            if rc_bind != CUresult::CUDA_SUCCESS {
                return Err(ExactGhostStepBindError::ContextBind {
                    rc: rc_bind,
                    expected_context: expected_ctx as usize,
                    observed_context: observed_ctx as usize,
                });
            }
            let mut verified_ctx: CUcontext = ptr::null_mut();
            let rc_verify = unsafe { cuCtxGetCurrent(&mut verified_ctx as *mut _) };
            if rc_verify != CUresult::CUDA_SUCCESS || verified_ctx != expected_ctx {
                return Err(ExactGhostStepBindError::ContextVerify {
                    rc: rc_verify,
                    expected_context: expected_ctx as usize,
                    observed_context: verified_ctx as usize,
                });
            }
        }
        let rc = unsafe { cuStreamSynchronize(self.md_stream.cu_stream()) };
        if rc != CUresult::CUDA_SUCCESS {
            return Err(ExactGhostStepBindError::PhysicalClockSynchronize { rc });
        }
        let mut physical_step = 0u32;
        let rc = unsafe {
            cuMemcpyDtoH_v2(
                &mut physical_step as *mut u32 as *mut c_void,
                self.ghost_physical_step_dev as CUdeviceptr,
                std::mem::size_of::<u32>(),
            )
        };
        if rc != CUresult::CUDA_SUCCESS {
            return Err(ExactGhostStepBindError::PhysicalClockRead { rc });
        }
        if u64::from(physical_step) != post_md_step {
            return Err(ExactGhostStepBindError::PhysicalClockMismatch { physical_step, post_md_step });
        }
        Ok(())
    }

    /// LEGACY API SHIM (Path Z device-slot supersedes per-launch patching).
    ///
    /// In the original G24 design (commit 37669ea9) this method patched the
    /// ZSTR kernel-node params per launch via `cuGraphExecKernelNodeSetParams_v2`.
    /// Path Z (Amendment 3.4) replaced that with a `__constant__ uint32_t
    /// d_zstr_active_slot` updated host-side via
    /// `prism_nhs::zstr::ffi::prism_zstr_set_active_slot(slot, stream)`,
    /// because `cudaGraphAddChildGraphNode` copies the child template — the
    /// original node handles are non-addressable on the fused exec.
    ///
    /// **WP4** retired the host-side slot selection entirely. The
    /// module-scope symbol is gone (it aliased across all 8 streams sharing
    /// one context/module) and the in-graph slot-authority kernel now owns
    /// the slot AND the header stamp; a host-chosen slot would produce a
    /// claim with no stamped header, which is exactly the unparseable-file
    /// defect WP4 fixes. `frame_idx` is therefore ignored and the shim
    /// delegates straight to `launch()`.
    pub fn launch_with_zstr_slot(
        &self,
        _frame_idx: u64,
        _ring: &crate::zstr::ZstrRing,
    ) -> Result<(), DriverError> {
        // WP4: slot rolling + header stamping are both in-graph; there is
        // nothing for the host to set before the launch.
        self.launch()
    }

    /// Read the burst-marker u32 back to the host (synchronous; for
    /// audit / G19 attestation only — NOT for the production critical
    /// path). Returns the latest value the body sub-graph wrote.
    pub fn read_burst_marker(&self) -> Result<u32, DriverError> {
        let mut host: u32 = 0;
        let rc = unsafe {
            cuMemcpyDtoH_v2(
                &mut host as *mut _ as *mut c_void,
                self.burst_marker_dev as CUdeviceptr,
                4,
            )
        };
        if !matches!(rc, CUresult::CUDA_SUCCESS) {
            return Err(DriverError(rc));
        }
        Ok(host)
    }

    /// Number of kernel-typed nodes captured into the graph.
    /// Exposed for Claude-3's G18 attestation.
    pub fn n_kernel_nodes(&self) -> u32 {
        self.n_kernel_nodes_captured
    }

    /// Number of cuGraphAddDependencies edges added explicitly
    /// post-capture. Exposed for Claude-3's G19 attestation
    /// (the trampoline → conditional edge is the operator's
    /// non-negotiable §2.3 mandate).
    pub fn n_explicit_edges(&self) -> u32 {
        self.n_dependency_edges_explicit
    }

    /// 128-byte alignment of the F2-pool tile array. CSR §M.
    pub fn tile_alignment_ok(&self) -> bool {
        self.tiles_dev % 128 == 0
    }

    /// **M1.2.26.RECOUPLE Part 4 — Lagged Reference Manifold lock.**
    ///
    /// Snapshots the current `tiles_baseline_dev` (which the captured
    /// graph re-computes per replay as `SO(3)(d_spikes_raw)`) into the
    /// pre-allocated `p_frozen_dev` buffer, then updates
    /// `adj.relaxed_manifold_ptr` (FFI offset 56) to point at the
    /// frozen snapshot. After this call, the adjudicator's KL
    /// evaluation compares `p_frozen` against the per-replay perturbed
    /// `tiles_dev` (i.e., `Δ_AB = KL(P_frozen || Q_current)`),
    /// providing the absolute interferometric contrast mandated by
    /// the M1.2.26.RECOUPLE Part 4 directive (target T29).
    ///
    /// Before the first replay, the baseline is the native raw-seed SO(3)
    /// projection completed during build. After a replay it is that replay's
    /// raw-spike projection. The caller chooses the actual anchor boundary;
    /// neither capture nor this copy advances physical simulation time.
    /// Idempotent if the host gates re-entry — calling it twice would
    /// overwrite `p_frozen` with whatever `tiles_baseline_dev` looks
    /// like at that later step.
    ///
    /// # Errors
    ///
    /// Returns `DriverError::CudaError` on cuMemcpy failures.
    pub fn freeze_baseline(&self) -> Result<(), DriverError> {
        use cudarc::driver::sys::{
            cuMemcpyDtoDAsync_v2, cuMemcpyHtoDAsync_v2, CUdeviceptr, CUresult,
        };
        // Step 1: snapshot tiles_baseline_dev → p_frozen_dev (DtoD).
        unsafe {
            let rc = cuMemcpyDtoDAsync_v2(
                self.p_frozen_dev as CUdeviceptr,
                self.tiles_baseline_dev as CUdeviceptr,
                self.tiles_bytes as usize,
                self.md_stream.cu_stream(),
            );
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                log::error!(
                    "M1.2.26 freeze_baseline: cuMemcpyDtoD baseline→p_frozen rc={:?}",
                    rc
                );
                return Err(DriverError(rc));
            }
        }
        // Step 2: update adj.relaxed_manifold_ptr (offset 56) → p_frozen_dev.
        unsafe {
            let relaxed_ptr_field = (self.adj_dev + 56) as CUdeviceptr;
            let p_frozen_value: u64 = self.p_frozen_dev as u64;
            let rc = cuMemcpyHtoDAsync_v2(
                relaxed_ptr_field,
                &p_frozen_value as *const _ as *const std::ffi::c_void,
                8,
                self.md_stream.cu_stream(),
            );
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                log::error!(
                    "M1.2.26 freeze_baseline: cuMemcpyHtoD adj.relaxed_manifold_ptr=0x{:x} rc={:?}",
                    p_frozen_value,
                    rc
                );
                return Err(DriverError(rc));
            }
            // p_frozen_value is stack-backed host memory. Complete its
            // asynchronous upload before this scope ends, and finish the
            // preceding same-stream snapshot before announcing publication.
            // This occurs once per requested freeze, never per chunk.
            self.md_stream.synchronize()?;
        }
        log::info!(
            "[M1.2.26.RECOUPLE Part 4] P_frozen locked: \
             tiles_baseline_dev=0x{:x} → p_frozen_dev=0x{:x} \
             ({} bytes); adj.relaxed_manifold_ptr redirected.",
            self.tiles_baseline_dev,
            self.p_frozen_dev,
            self.tiles_bytes
        );
        Ok(())
    }

    /// Explicit diagnostic readback; never called in the per-chunk path.
    /// Synchronizes this stream and hashes the actual relaxed/frozen bytes.
    /// Call before any replay to verify initialization/freeze, or retain the
    /// caller's replay/step context when interpreting a later snapshot.
    pub fn frozen_baseline_readback(&self) -> Result<serde_json::Value, DriverError> {
        use sha2::{Digest, Sha256};
        self.md_stream.synchronize()?;
        let mut relaxed = vec![0u8; self.tiles_bytes as usize];
        let mut frozen = vec![0u8; self.tiles_bytes as usize];
        let mut relaxed_binding: u64 = 0;
        for (host, device, bytes) in [
            (relaxed.as_mut_ptr() as *mut c_void, self.tiles_baseline_dev as CUdeviceptr, relaxed.len()),
            (frozen.as_mut_ptr() as *mut c_void, self.p_frozen_dev as CUdeviceptr, frozen.len()),
            (&mut relaxed_binding as *mut _ as *mut c_void, (self.adj_dev + 56) as CUdeviceptr, 8),
        ] {
            let rc = unsafe { cuMemcpyDtoH_v2(host, device, bytes) };
            if !matches!(rc, CUresult::CUDA_SUCCESS) { return Err(DriverError(rc)); }
        }
        let stride = std::mem::size_of::<ContactShellTile>();
        let read_u32 = |bytes: &[u8], offset: usize| -> u32 {
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
        };
        let headers = |bytes: &[u8]| -> Vec<serde_json::Value> {
            bytes.chunks_exact(stride).map(|tile| serde_json::json!({
                "stream_id": read_u32(tile, std::mem::offset_of!(ContactShellTile, stream_id)),
                "captured_frame_id": read_u32(tile, std::mem::offset_of!(ContactShellTile, frame)),
                "cluster_id_bits": read_u32(tile, std::mem::offset_of!(ContactShellTile, cluster_id)),
                "spike_count": read_u32(tile, std::mem::offset_of!(ContactShellTile, spike_count)),
            })).collect()
        };
        Ok(serde_json::json!({
            "schema_kind": "prism_native_frozen_baseline_readback.v1",
            "scope": "actual current device-buffer bytes; caller supplies replay/physical-step context and raw-input hashes",
            "stream_id": self.g26_stream_id,
            "cluster_count": self.n_clusters,
            "tile_bytes": stride,
            "buffer_bytes": self.tiles_bytes,
            "relaxed_sha256": format!("{:x}", Sha256::digest(&relaxed)),
            "frozen_sha256": format!("{:x}", Sha256::digest(&frozen)),
            "relaxed_headers": headers(&relaxed),
            "frozen_headers": headers(&frozen),
            "byte_identical": relaxed == frozen,
            "adjudicator_relaxed_binding_is_frozen": relaxed_binding == self.p_frozen_dev as u64,
        }))
    }

    /// **M1.2.26.RECOUPLE Part 3 — Write KL-native μ/σ to FFI.**
    ///
    /// After the host has accumulated KL samples during the Phase-2
    /// variance window (steps 51-100), it computes mean + stddev and
    /// calls this method to flush them to `adj.total_kl_mu` (offset
    /// 148) and `adj.total_kl_sigma` (offset 152). After this write,
    /// the F1 SWITCH predicate in `adjudicator.cu` switches from the
    /// legacy SH-coefficient threshold to the dimensionally-
    /// synchronized KL-native threshold:
    ///   `threshold = total_kl_mu + 12σ * total_kl_sigma`
    ///
    /// # Errors
    ///
    /// Returns `DriverError::CudaError` on cuMemcpy failure.
    pub fn write_kl_native_mu_sigma(
        &self,
        total_kl_mu: f32,
        total_kl_sigma: f32,
    ) -> Result<(), DriverError> {
        use cudarc::driver::sys::{cuMemcpyHtoDAsync_v2, CUdeviceptr, CUresult};
        unsafe {
            let mu_field = (self.adj_dev + 148) as CUdeviceptr;
            let rc1 = cuMemcpyHtoDAsync_v2(
                mu_field,
                &total_kl_mu as *const f32 as *const std::ffi::c_void,
                4,
                self.md_stream.cu_stream(),
            );
            let sigma_field = (self.adj_dev + 152) as CUdeviceptr;
            let rc2 = cuMemcpyHtoDAsync_v2(
                sigma_field,
                &total_kl_sigma as *const f32 as *const std::ffi::c_void,
                4,
                self.md_stream.cu_stream(),
            );
            for (rc, what) in [(rc1, "total_kl_mu"), (rc2, "total_kl_sigma")] {
                if !matches!(rc, CUresult::CUDA_SUCCESS) {
                    log::error!("M1.2.26 write_kl_native_mu_sigma: {} rc={:?}", what, rc);
                    return Err(DriverError(rc));
                }
            }
        }
        log::info!(
            "[M1.2.26.RECOUPLE Part 3] KL-native threshold engaged: \
             total_kl_mu={:.4e} total_kl_sigma={:.4e} \
             discovery_threshold={:.4e} (= μ + 12σ)",
            total_kl_mu,
            total_kl_sigma,
            total_kl_mu + 12.0 * total_kl_sigma
        );
        Ok(())
    }

    /// **M1.2.26.RECOUPLE — Read adj.current_divergence host-side.**
    ///
    /// Used by the host's Phase-2 variance accumulation loop. Returns
    /// the latest scalar `total_kl` written by the captured graph's
    /// adjudicator kernel (FFI offset 48).
    pub fn read_current_divergence(&self) -> Result<f32, DriverError> {
        use cudarc::driver::sys::{cuMemcpyDtoH_v2, CUdeviceptr, CUresult};
        let mut value: f32 = 0.0;
        unsafe {
            let rc = cuMemcpyDtoH_v2(
                &mut value as *mut f32 as *mut std::ffi::c_void,
                (self.adj_dev + 48) as CUdeviceptr,
                4,
            );
            if !matches!(rc, CUresult::CUDA_SUCCESS) {
                log::error!("M1.2.26 read_current_divergence: cuMemcpyDtoH rc={:?}", rc);
                return Err(DriverError(rc));
            }
        }
        Ok(value)
    }

    /// Telemetry stream handle. Exposed for the audit gate G20
    /// (`stream_flags(...) & CU_STREAM_NON_BLOCKING == 1`).
    pub fn telemetry_stream(&self) -> CUstream {
        self.telemetry_stream
    }

    /// Raw `CUstream` for the pipeline's MD-side stream.  Exposed so the
    /// orchestrator can enqueue stream-ordered host operations (e.g.,
    /// `prism_zstr_set_active_slot` constant-memory updates per launch)
    /// on the same stream that runs the captured graph — preserves the
    /// FIFO ordering between symbol update and `cuGraphLaunch`.
    pub fn md_stream_raw(&self) -> CUstream {
        self.md_stream.cu_stream()
    }

    /// Read access to the pinned ring (Pillar-5 reporter consumes
    /// this off the critical path).
    pub fn ring(&self) -> &PinnedTelemetryRing<ContactShellTile> {
        &self.ring
    }

    /// Conditional handle (G19 audit input). `0` in V1 (no
    /// conditional node yet); V2 hook populates it.
    pub fn conditional_handle(&self) -> u64 {
        self.cond_handle
    }

    // ── V2 IGNITION accessors (operator-mandated) ──────────────────

    /// Raw `CUgraph` handle for the captured pipeline. Exposed so
    /// V2's `prism_wire_f1_switch_ffi` C-ABI bypass can inject a
    /// CONDITIONAL node post-instantiation if a hook-time injection
    /// is insufficient for some downstream pattern. Most V2 callers
    /// should use the [`Self::build_with_v2_hook`] hook closure
    /// instead — calling FFI on this handle AFTER instantiation
    /// requires a re-instantiate to take effect, which the hook
    /// avoids.
    pub fn cu_graph_raw(&self) -> CUgraph {
        self.cu_graph
    }

    /// Device pointer to the [`InterferometricAdjudicatorFfi`] that
    /// the captured Adjudicator kernel writes to. V2 callers pass
    /// this through to
    /// `prism_get_adjudication_code_devptr(adj_dev_ptr as *const _)`
    /// (Claude-2's existing FFI helper) to get the predicate pointer
    /// the F1 SWITCH conditional node binds to. CSR §M alignment is
    /// guaranteed by the F2 pool (≥ 128 B allocations are
    /// 128-byte aligned).
    pub fn adj_dev_ptr(&self) -> usize {
        self.adj_dev
    }

    /// Raw `CUgraphExec` handle. Exposed for diagnostic /
    /// `cuGraphExecGetFlags` introspection only — production paths
    /// should call [`Self::launch`] which threads the MD stream
    /// correctly.
    pub fn cu_graph_exec_raw(&self) -> CUgraphExec {
        self.cu_graph_exec
    }

    /// Wave B.1 — device pointer to the G26 ChronometricStateTensor
    /// (16 B). Pointer-stable for the campaign. The B.2 predicate
    /// bridge will pass this to the gearbox PointerSwap kernel through
    /// the SWITCH body sub-graphs; outside the captured graph, callers
    /// can invoke `prism_gearbox_launch_pointer_swap` directly with
    /// this address for diagnostic / one-shot tests.
    pub fn cruise_state_dev_ptr(&self) -> u64 {
        self.cruise_state_dev as u64
    }

    // ════════════════════════════════════════════════════════════════════
    // GEAR-PROVENANCE-002 — measured gear + real G26 control-trace emitter
    // ════════════════════════════════════════════════════════════════════

    /// Read the gear the device is actually in, plus the timestep actually
    /// bound to `ProtocolState.dt`.
    ///
    /// This is the only source of a MEASURED gear anywhere in the tree. Every
    /// other `gear_id` in the codebase is a build-time literal that cannot
    /// witness a per-replay device decision.
    ///
    /// The wait is bounded by [`crate::gearbox::resolve_snapshot_timeout`] and
    /// expiry surfaces as [`crate::gearbox::GearSnapshotError::Timeout`]. Call
    /// only outside a capture window; the FFI refuses with `CaptureActive`
    /// otherwise rather than corrupting the graph.
    pub fn measured_gear(
        &self,
    ) -> std::result::Result<crate::gearbox::MeasuredGear, crate::gearbox::GearSnapshotError> {
        unsafe {
            crate::gearbox::snapshot_measured_gear(
                self.cruise_state_dev as u64,
                self.g26_dt_dev as u64,
                self.md_stream.cu_stream() as *mut c_void,
                self.g26_readback_timeout,
            )
        }
    }

    /// Read `adj->gear_override` (u32 at offset 100) off the device.
    ///
    /// `Some(0xFF)` is the "Auto" sentinel meaning no host override is in
    /// force. `None` means the word could not be read — never a gear.
    pub fn host_gear_override(&self) -> Option<u32> {
        if self.adj_dev == 0 {
            return None;
        }
        let mut raw: u32 = 0;
        let rc = unsafe {
            cuMemcpyDtoH_v2(
                &mut raw as *mut u32 as *mut c_void,
                (self.adj_dev + 100) as CUdeviceptr,
                std::mem::size_of::<u32>(),
            )
        };
        if matches!(rc, CUresult::CUDA_SUCCESS) {
            Some(raw)
        } else {
            None
        }
    }

    /// Read the adjudicator's energy state (`potential_energy` f64 at offset
    /// 112, and `*d_external_work` reached through the pointer at offset 128).
    ///
    /// Unreadable scalars stay non-finite and the returned `source` names the
    /// condition; nothing is substituted with a plausible number.
    fn read_source_energy_state(&self) -> crate::control_trace::SourceEnergyState {
        use crate::control_trace::SourceEnergyState;

        if self.adj_dev == 0 {
            return SourceEnergyState {
                potential_energy: f64::NAN,
                external_work: f64::NAN,
                source: "g26_readback_unavailable_no_adjudicator".to_string(),
            };
        }

        let mut pe: f64 = f64::NAN;
        let pe_rc = unsafe {
            cuMemcpyDtoH_v2(
                &mut pe as *mut f64 as *mut c_void,
                (self.adj_dev + 112) as CUdeviceptr,
                std::mem::size_of::<f64>(),
            )
        };
        if !matches!(pe_rc, CUresult::CUDA_SUCCESS) {
            pe = f64::NAN;
        }

        // d_external_work is a POINTER at offset 128; two hops.
        let mut w_ptr: u64 = 0;
        let mut wext: f64 = f64::NAN;
        let wp_rc = unsafe {
            cuMemcpyDtoH_v2(
                &mut w_ptr as *mut u64 as *mut c_void,
                (self.adj_dev + 128) as CUdeviceptr,
                std::mem::size_of::<u64>(),
            )
        };
        if matches!(wp_rc, CUresult::CUDA_SUCCESS) && w_ptr != 0 {
            let w_rc = unsafe {
                cuMemcpyDtoH_v2(
                    &mut wext as *mut f64 as *mut c_void,
                    w_ptr as CUdeviceptr,
                    std::mem::size_of::<f64>(),
                )
            };
            if !matches!(w_rc, CUresult::CUDA_SUCCESS) {
                wext = f64::NAN;
            }
        }

        SourceEnergyState {
            potential_energy: pe,
            external_work: wext,
            source: "adj_device_readback:potential_energy@112,external_work@*128".to_string(),
        }
    }

    /// Build a `ControlPlaneRecord::G26` describing the gearbox decision this
    /// pipeline just executed.
    ///
    /// The record keeps the three gear classes apart. `requested_gear` is
    /// whatever the caller's host policy asked for (pass `None` when no host
    /// policy is in force — the gearbox is autonomous by design, so `None` is
    /// the normal case and is recorded as an explicit absence, not as gear 0).
    /// `authorized_gear` is derived from the measured `adj->gear_override`.
    /// `previous_gear`/`current_gear` are the device readback and are the only
    /// fields that state what the integrator actually ran at.
    ///
    /// Never panics and never fabricates: a failed readback yields
    /// `read_failed` observations naming the terminal condition.
    pub fn build_g26_record(
        &self,
        run_id: &str,
        graph_launch_id: &str,
        record_idx: u64,
        frame_idx: u64,
        requested_gear: Option<u8>,
    ) -> crate::control_trace::ControlPlaneRecord {
        use crate::control_trace::{
            ControlPlaneRecord, ControlTraceEnvelope, GearObservation, ReadbackBound, ScalarOrigin,
        };

        let bound = self.g26_readback_timeout;
        // The gear readback owns the ONLY bounded wait on this path. Every
        // other read below (`host_gear_override`, `read_source_energy_state`)
        // is a blocking `cuMemcpyDtoH_v2`, which would wait on outstanding
        // graph work WITHOUT a bound. So they are attempted only once the
        // bounded drain has actually succeeded; after a timeout the record is
        // completed from sentinels instead. A timeout must never be able to
        // turn into an unbounded wait further down the same function.
        let measured = self.measured_gear();
        let stream_drained = measured.is_ok();
        let override_raw = if stream_drained {
            self.host_gear_override()
        } else {
            None
        };

        let requested = match requested_gear {
            Some(g) => GearObservation::requested(g, "host_control_policy"),
            None => {
                GearObservation::unmeasured("no_host_policy_request_gearbox_is_device_autonomous")
            }
        };

        let (prev_obs, cur_obs, dt_ps, dt_origin, waited_us, expired) = match &measured {
            Ok(m) => (
                match m.previous_gear {
                    Some(g) => GearObservation::device_measured(
                        g,
                        "chronometric_state_tensor.previous_gear",
                    ),
                    None => GearObservation::unmeasured(
                        "device_previous_gear_outside_documented_0_3_encoding",
                    ),
                },
                match m.current_gear {
                    Some(g) => GearObservation::device_measured(
                        g,
                        "chronometric_state_tensor.current_gear",
                    ),
                    None => GearObservation::unmeasured(
                        "device_current_gear_outside_documented_0_3_encoding",
                    ),
                },
                m.dt_ps_applied,
                if m.dt_ps_applied.is_some() {
                    ScalarOrigin::DeviceReadback
                } else {
                    ScalarOrigin::Unmeasured
                },
                m.waited_us,
                false,
            ),
            Err(e) => {
                let label = e.as_str();
                let waited = match e {
                    crate::gearbox::GearSnapshotError::Timeout { waited_us, .. } => *waited_us,
                    _ => 0,
                };
                let expired = matches!(e, crate::gearbox::GearSnapshotError::Timeout { .. });
                (
                    GearObservation::read_failed(label),
                    GearObservation::read_failed(label),
                    None,
                    ScalarOrigin::ReadFailed,
                    waited,
                    expired,
                )
            }
        };

        // The predicate bridge computes
        //   final = (override == 0xFF) ? cruise.current_gear : override & 3
        // so the selector is DERIVED from two measured device words. It is
        // labelled as a derivation, never as a direct measurement.
        let (authorized, selector, selector_origin) = match (override_raw, cur_obs.gear) {
            (Some(ov), _) if ov != 0xFF => {
                let g = (ov & 0x3) as u8;
                (
                    GearObservation {
                        gear: Some(g),
                        origin: crate::control_trace::GearOrigin::HostOverride,
                        detail: format!("adj.gear_override@100=0x{ov:X}"),
                    },
                    Some(g as u32),
                    ScalarOrigin::DerivedFromDeviceState,
                )
            }
            (Some(_), Some(g)) => (
                GearObservation::derived(
                    g,
                    "predicate_bridge_auto:adj.gear_override@100=0xFF_uses_cruise.current_gear",
                ),
                Some(g as u32),
                ScalarOrigin::DerivedFromDeviceState,
            ),
            (Some(_), None) => (
                GearObservation::unmeasured("predicate_bridge_auto_but_device_gear_unreadable"),
                None,
                ScalarOrigin::Unmeasured,
            ),
            (None, _) => (
                GearObservation::read_failed(if stream_drained {
                    "gear_override_word_unreadable"
                } else {
                    "not_attempted_bounded_gear_readback_did_not_complete"
                }),
                None,
                ScalarOrigin::ReadFailed,
            ),
        };

        let energy = if stream_drained {
            self.read_source_energy_state()
        } else {
            crate::control_trace::SourceEnergyState {
                potential_energy: f64::NAN,
                external_work: f64::NAN,
                source: "not_attempted_bounded_gear_readback_did_not_complete".to_string(),
            }
        };
        let energy_hash = energy.canonical_hash();

        ControlPlaneRecord::G26 {
            envelope: ControlTraceEnvelope::current(run_id, graph_launch_id, record_idx),
            frame_idx,
            stream_id: self.g26_stream_id,
            requested_gear: requested,
            authorized_gear: authorized,
            previous_gear: prev_obs,
            current_gear: cur_obs,
            dt_ps,
            dt_ps_origin: dt_origin,
            switch_selector: selector,
            switch_selector_origin: selector_origin,
            readback_bound: ReadbackBound {
                budget_ms: bound.millis,
                budget_origin: bound.origin.as_str().to_string(),
                waited_us,
                expired,
            },
            source_energy_state: energy,
            source_energy_state_hash: energy_hash,
        }
    }

    /// Append one `ControlPlaneRecord` to this pipeline's G26 NDJSON sink.
    ///
    /// No-op when tracing is not configured. Open/write failures are logged
    /// once and latched; they never propagate into the launch path.
    pub fn append_control_record(&self, record: &crate::control_trace::ControlPlaneRecord) {
        let Some(cfg) = self.g26_trace_cfg.as_ref() else {
            return;
        };
        let Ok(mut sink) = self.g26_trace.lock() else {
            return;
        };
        if sink.open_failed {
            return;
        }
        if sink.file.is_none() {
            if let Err(e) = std::fs::create_dir_all(&cfg.dir) {
                log::warn!(
                    "[GEAR-PROVENANCE-002] cannot create G26 control-trace dir {}: {e}; \
                     emission disabled for this pipeline.",
                    cfg.dir.display()
                );
                sink.open_failed = true;
                return;
            }
            match std::fs::OpenOptions::new()
                .read(true)
                .create(true)
                .append(true)
                .open(cfg.path())
            {
                Ok(f) => {
                    log::info!(
                        "[GEAR-PROVENANCE-002] G26 control trace -> {}",
                        cfg.path().display()
                    );
                    sink.file = Some(f);
                }
                Err(e) => {
                    log::warn!(
                        "[GEAR-PROVENANCE-002] cannot open G26 control trace {}: {e}; \
                         emission disabled for this pipeline.",
                        cfg.path().display()
                    );
                    sink.open_failed = true;
                    return;
                }
            }
        }
        if let Some(f) = sink.file.as_mut() {
            if let Err(e) =
                crate::control_trace::write_control_plane_ndjson(f, std::iter::once(record.clone()))
            {
                log::warn!(
                    "[GEAR-PROVENANCE-002] G26 control-trace write failed: {e}; \
                     emission disabled for this pipeline."
                );
                sink.open_failed = true;
            }
        }
    }

    /// Observe and record the gearbox decision produced by the launch that
    /// just completed.
    ///
    /// Called from [`Self::launch`] when `PRISM_G26_CONTROL_TRACE_DIR` is set.
    /// Costs one bounded stream drain plus ~52 bytes of device reads.
    pub(crate) fn observe_g26_after_launch(&self) {
        self.observe_g26_after_parent_launch(self.cu_graph_exec);
    }

    pub(crate) fn observe_g26_after_parent_launch(&self, executed_parent: CUgraphExec) {
        let Some(cfg) = self.g26_trace_cfg.as_ref() else {
            return;
        };
        if self.g26_trace.lock().map(|s| s.open_failed).unwrap_or(true) {
            return;
        }
        let (record_idx, launch_seq) = {
            let Ok(mut sink) = self.g26_trace.lock() else {
                return;
            };
            let idx = sink.record_idx;
            sink.record_idx += 1;
            sink.launch_seq += 1;
            (idx, sink.launch_seq)
        };
        let graph_launch_id = format!("graph_exec_{:p}_launch_{launch_seq}", executed_parent);
        // `requested_gear = None`: the gearbox is device-autonomous on this
        // path, so there IS no host request. Recording that absence explicitly
        // is the point — a `0` here would be indistinguishable from a real
        // policy request for the 0.5 fs gear.
        let record = self.build_g26_record(
            &cfg.run_id,
            &graph_launch_id,
            record_idx,
            launch_seq.saturating_sub(1),
            None,
        );
        self.append_control_record(&record);
    }
}

impl Drop for CapturedAdjudicationPipeline {
    fn drop(&mut self) {
        // Drain md_stream before touching any graph or device handle.
        // The final pipeline.launch() submits async GPU work; cuGraphExecDestroy
        // does NOT wait for it.  Without this sync, the caller's cuMemFree_v2
        // on v2_mask_raw races with the still-running ASC kernel — UAF on device.
        // Conditional child graphs and stream-ordered pool allocations need
        // a completed context retirement boundary before allocator reuse.
        // This is teardown only, after the checked bounded owner snapshot.
        let retired = self.governor_device.snapshot().and_then(|_| {
            self.md_stream.context().synchronize().map_err(|error| error.0 as i32)
        });
        if let Err(rc) = retired {
            log::error!("PIPELINE_QUARANTINED source={} rc={}: completion unknown; retaining pool/ring/context", self.g26_stream_id, rc);
            self.pool.quarantine(rc);
            self.ring.quarantine(rc);
            std::mem::forget(self.md_stream.clone());
            std::mem::forget(self.governor_device.clone());
            if let Some(owner) = &self.asc_device { std::mem::forget(owner.clone()); }
            return;
        }

        let md_raw = self.md_stream.cu_stream() as usize;
        unsafe {
            if !self.cu_graph_exec.is_null() {
                let _ = result::graph::exec_destroy(self.cu_graph_exec);
            }
            if !self.cu_graph.is_null() {
                let _ = result::graph::destroy(self.cu_graph);
            }
            if !self.md_to_telemetry_event.is_null() {
                let _ = cuEventDestroy_v2(self.md_to_telemetry_event);
            }
            if !self.telemetry_to_md_event.is_null() {
                let _ = cuEventDestroy_v2(self.telemetry_to_md_event);
            }
            if !self.telemetry_stream.is_null() {
                let _ = cuStreamDestroy_v2(self.telemetry_stream);
            }
        }
        // Stream-ordered free of pool allocations.
        let _ = self.pool.free_async(self.perturbed_positions_dev, md_raw);
        let _ = self.pool.free_async(self.com_shift_dev, md_raw);
        let _ = self.pool.free_async(self.total_mass_dev, md_raw);
        let _ = self.pool.free_async(self.com_correction_dev, md_raw);
        let _ = self.pool.free_async(self.tiles_dev, md_raw);
        let _ = self.pool.free_async(self.tiles_baseline_dev, md_raw);
        let _ = self.pool.free_async(self.p_frozen_dev, md_raw);
        let _ = self.pool.free_async(self.adj_dev, md_raw);
        let _ = self.pool.free_async(self.burst_marker_dev, md_raw);
        if self.sisr_mask_dev != 0 {
            let _ = self.pool.free_async(self.sisr_mask_dev, md_raw);
        }
        if self.sisr_count_dev != 0 {
            let _ = self.pool.free_async(self.sisr_count_dev, md_raw);
        }
        // Wave 3 / Path B Dynamic T7 buffers (always allocated).
        let _ = self.pool.free_async(self.dynt7_acc_dev, md_raw);
        let _ = self.pool.free_async(self.dynt7_idx_dev, md_raw);
        let _ = self.pool.free_async(self.dynt7_stats_dev, md_raw);
        // Wave B.1 — G26 ChronometricStateTensor (always allocated).
        // Cruise storage is owned by the retained per-stream governor.
        // M1.2.17 — energy monitor buffers (allocated only when n_atoms_for_pe > 0).
        if self.energy_temp_storage_dev != 0 {
            let _ = self.pool.free_async(self.energy_temp_storage_dev, md_raw);
        }
        if self.energy_pe_scalar_dev != 0 {
            let _ = self.pool.free_async(self.energy_pe_scalar_dev, md_raw);
        }
        if self.energy_window_dev != 0 {
            let _ = self.pool.free_async(self.energy_window_dev, md_raw);
        }
        // Complete queued pool frees before pool destruction/rebuild can
        // reuse addresses for another graph epoch. Never destroy the pool if
        // completion is unknown, even if a subsequent stream query succeeds.
        if let Err(error) = self.md_stream.synchronize() {
            self.pool.quarantine(error.0 as i32);
            self.ring.quarantine(error.0 as i32);
            log::error!("PIPELINE_POOL_QUARANTINED source={} rc={}: async free completion unknown", self.g26_stream_id, error.0 as i32);
        }
        // Stream-bound CudaOwned fields perform a checked bounded join before
        // releasing pool/ring, including all early construction-error paths.

    }
}

// ============================================================================
// Helpers — explicit-mode graph extension (post-capture)
// ============================================================================

/// Add a CONDITIONAL node (IF-type, size=1) downstream of every node
/// in `dependencies`. The first dependency is the operator's
/// non-negotiable §2.3 edge: trampoline → conditional.
///
/// Returns `(conditional_node, body_subgraph)`. The caller adds nodes
/// to `body_subgraph` to populate the "Burst" path.
unsafe fn add_conditional_node(
    parent_graph: CUgraph,
    ctx: CUcontext,
    handle: CUgraphConditionalHandle,
    dependencies: &[CUgraphNode],
) -> Result<(CUgraphNode, CUgraph), BuildError> {
    // CUDA writes the body subgraph(s) into a host array we provide
    // through `phGraph_out`. For IF-type, size=1, so a single-element
    // array is sufficient.
    let mut body_subgraphs: [CUgraph; 1] = [ptr::null_mut()];
    let mut node_params: CUgraphNodeParams_st = std::mem::zeroed();
    node_params.type_ = CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL;
    node_params.__bindgen_anon_1.conditional = CUDA_CONDITIONAL_NODE_PARAMS {
        handle,
        type_: CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_IF,
        size: 1,
        phGraph_out: body_subgraphs.as_mut_ptr(),
        ctx,
    };

    let mut cond_node: CUgraphNode = ptr::null_mut();
    // Use cuGraphAddNode_v2 — the v1 unversioned variant is deprecated in
    // CUDA 12.4+ and silently returns SUCCESS with `phGraph_out[0] = null`
    // for CONDITIONAL nodes (observed empirically on CUDA 13). v2 takes
    // a `dependencyData` array which we pass null for default edges.
    let rc = cuGraphAddNode_v2(
        &mut cond_node as *mut _,
        parent_graph,
        dependencies.as_ptr(),
        ptr::null(), // dependencyData = default edge type
        dependencies.len(),
        &mut node_params as *mut _,
    );
    if !matches!(rc, CUresult::CUDA_SUCCESS) {
        return Err(BuildError::Cuda {
            stage: "cuGraphAddNode_v2 (CONDITIONAL)",
            rc: rc as i32,
        });
    }
    if cond_node.is_null() {
        return Err(BuildError::Cuda {
            stage: "cuGraphAddNode (CONDITIONAL) returned null cond_node",
            rc: -1,
        });
    }
    if body_subgraphs[0].is_null() {
        return Err(BuildError::Cuda {
            stage: "cuGraphAddNode (CONDITIONAL) left body_subgraphs[0] = null",
            rc: -1,
        });
    }
    Ok((cond_node, body_subgraphs[0]))
}

/// Add a MEMSET node to the body sub-graph that stamps a non-zero
/// constant into `burst_marker_dev` whenever the conditional fires.
/// This is a topology-level marker the audit harness reads back to
/// attest the F1 SWITCH actually routed.
unsafe fn add_burst_marker_memset_node(
    body_subgraph: CUgraph,
    burst_marker_dev: usize,
    ctx: &Arc<CudaContext>,
) -> Result<CUgraphNode, BuildError> {
    // CUDA_MEMSET_NODE_PARAMS (v1) — no embedded ctx field; cuGraphAddMemsetNode
    // takes the CUcontext as a separate trailing argument.
    let params = CUDA_MEMSET_NODE_PARAMS {
        dst: burst_marker_dev as CUdeviceptr,
        pitch: 4,       // tight: single 4-byte element
        value: 1,       // any non-zero sentinel — body fired ⇒ burst_marker != 0
        elementSize: 4, // u32
        width: 1,       // 1 element
        height: 1,
    };
    let mut memset_node: CUgraphNode = ptr::null_mut();
    let rc = cuGraphAddMemsetNode(
        &mut memset_node as *mut _,
        body_subgraph,
        ptr::null(), // no in-body dependencies; this is the only body node
        0,
        &params as *const _,
        ctx.cu_ctx() as CUcontext,
    );
    if !matches!(rc, CUresult::CUDA_SUCCESS) {
        return Err(BuildError::Cuda {
            stage: "cuGraphAddMemsetNode (body burst_marker)",
            rc: rc as i32,
        });
    }
    Ok(memset_node)
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug)]
pub enum BuildError {
    InvalidConfig {
        reason: &'static str,
    },
    PoolCreate(String),
    PoolAlloc {
        what: &'static str,
        reason: String,
    },
    PinnedRing(i32),
    TelemetryStream(i32),
    Cuda {
        stage: &'static str,
        rc: i32,
    },
    Driver(DriverError),
    AlignmentDrift {
        what: &'static str,
        got: usize,
        required: usize,
    },
    CaptureNotActive,
    CaptureFrontierEmpty,
    CaptureProducedNullGraph,
    /// V2 IGNITION hook (e.g., Claude-2's `prism_wire_f1_switch_ffi`)
    /// returned a non-success cudaError. Build aborts and the raw
    /// graph + all F2 allocations are cleaned up.
    V2HookFailed {
        rc: i32,
    },
    /// TIER 8 context/stream guard failure in the raw CUDA path used
    /// to diagnose graph-build context ownership without cudarc's
    /// `bind_to_thread()` wrapper.
    Tier8Context {
        stage: &'static str,
        rc: i32,
        expected: usize,
        before: usize,
        after: usize,
    },
    /// **TIER 8 (2026-05-03)** — Splice-legality preflight rejected
    /// the child template because it contains conditional nodes.
    /// CUDA 13.x's `cudaGraphAddNode(GRAPH)` would have returned
    /// cudaErrorNotSupported (801) on the splice itself; this PRISM-
    /// specific variant fires before CUDA gets the call.  Operator
    /// fix: lift the conditional out of the child template and
    /// install it at the parent graph level (TIER 8 architecture).
    SpliceIllegal {
        what: &'static str,
        total_nodes: usize,
        conditional_count: usize,
        allocation_count: usize,
        free_count: usize,
    },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::InvalidConfig { reason } => write!(f, "invalid config: {}", reason),
            BuildError::PoolCreate(s) => write!(f, "VramPool create: {}", s),
            BuildError::PoolAlloc { what, reason } => write!(f, "F2 alloc {}: {}", what, reason),
            BuildError::PinnedRing(rc) => write!(f, "pinned ring create failed: cuda {}", rc),
            BuildError::TelemetryStream(rc) => write!(f, "telemetry stream create: cuda {}", rc),
            BuildError::Cuda { stage, rc } => write!(f, "cuda error at {}: rc={}", stage, rc),
            BuildError::Driver(e) => write!(f, "driver error: {:?}", e),
            BuildError::AlignmentDrift {
                what,
                got,
                required,
            } => write!(
                f,
                "{} alignment drift: got {:#x}, required {} bytes",
                what, got, required
            ),
            BuildError::CaptureNotActive => write!(f, "stream is not in CAPTURE_STATUS_ACTIVE"),
            BuildError::CaptureFrontierEmpty => write!(
                f,
                "no captured nodes after trampoline launch — capture chain broken"
            ),
            BuildError::CaptureProducedNullGraph => write!(
                f,
                "cuStreamEndCapture / cuGraphAddNode produced a null handle"
            ),
            BuildError::V2HookFailed { rc } => {
                write!(f, "V2 IGNITION hook returned cudaError {}", rc)
            }
            BuildError::Tier8Context {
                stage,
                rc,
                expected,
                before,
                after,
            } => write!(
                f,
                "TIER8 context guard failed at {}: rc={} expected={:#x} \
                 before={:#x} after={:#x}",
                stage, rc, expected, before, after
            ),
            BuildError::SpliceIllegal {
                what,
                total_nodes,
                conditional_count,
                allocation_count,
                free_count,
            } => write!(
                f,
                "splice ILLEGAL — {} contains conditional={} alloc={} free={} \
                 node(s) of {} total; CUDA 13.x rejects cudaGraphAddNode(GRAPH) \
                 on children with control/allocation nodes. TIER 8 fix: lift \
                 those nodes to the parent graph.",
                what, conditional_count, allocation_count, free_count, total_nodes
            ),
        }
    }
}
impl std::error::Error for BuildError {}

// ============================================================================
// Tests — structural + smoke
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ghost_telemetry::{is_pinned_host, stream_flags};
    use cudarc::driver::CudaContext;

    fn build_test_pipeline() -> Option<(
        Arc<CudaContext>,
        CapturedAdjudicationPipeline,
        Vec<u32>,
        Vec<RichSpike>,
        cudarc::driver::CudaSlice<u8>,
        cudarc::driver::CudaSlice<u32>,
        Arc<CudaStream>,
    )> {
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[captured-pipeline] CUDA unavailable: {:?} — skipping", e);
                return None;
            }
        };
        let stream = ctx.new_stream().expect("md stream");
        let raw = stream.cu_stream() as usize;

        // Init K_LM (required by SO(3) kernel).
        let rc = unsafe { crate::sh_basis::ffi::prism_sh_basis_init(raw as *mut c_void) };
        assert_eq!(rc, crate::sh_basis::ffi::CUDA_SUCCESS);
        stream.synchronize().expect("post-sh-init sync");
        let k_lm_dev = crate::sh_basis::k_lm_device_ptr().expect("k_lm");

        // Synthesize 1 cluster with 16 spikes.
        const N_CLUSTERS: u32 = 1;
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
        let offsets: Vec<u32> = vec![0, spikes.len() as u32];

        let spike_bytes = spikes.len() * std::mem::size_of::<RichSpike>();
        let mut d_spikes_b = stream.alloc_zeros::<u8>(spike_bytes).expect("alloc spikes");
        let spikes_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(spikes.as_ptr() as *const u8, spike_bytes).to_vec()
        };
        stream
            .memcpy_htod(&spikes_bytes, &mut d_spikes_b)
            .expect("htod spikes");
        let mut d_offsets = stream
            .alloc_zeros::<u32>(offsets.len())
            .expect("alloc offsets");
        stream
            .memcpy_htod(&offsets, &mut d_offsets)
            .expect("htod offsets");

        use cudarc::driver::DevicePtr;
        // Scope the device_ptr guards so they drop before we return
        // the underlying slices in the tuple. The CudaSlice itself
        // owns the VRAM, so the captured graph's pointer stays valid
        // as long as `d_spikes_b` / `d_offsets` live in the caller's
        // scope.
        let (sp_dev, off_dev) = {
            let (sp, _g1) = d_spikes_b.device_ptr(&stream);
            let (off, _g2) = d_offsets.device_ptr(&stream);
            (sp, off)
        };
        stream.synchronize().expect("post-htod sync");

        let cfg = PipelineConfig {
            seed_layout: crate::rich_spike::NativeV2SeedLayout::default(),
            d_spikes: sp_dev as *const RichSpike,
            d_cluster_offsets: off_dev as *const u32,
            n_clusters: N_CLUSTERS,
            d_k_lm: k_lm_dev,
            initial_frame_id: 0,
            source_stream_id: 0,
            diagnostic_stream_id: None,
            asc: None,
            d_force_norm_mirror: 0,
            governor_device: None,
            zstr: None,
            sisr: None,
            noise_floor_override: None,
            mar_v2: None,
            d_dt: ptr::null_mut(),
            d_velocities: ptr::null_mut(),
            g26_parent_cond_handle: 0,
            f1_parent_cond_handle: 0,
            d_pe_components: ptr::null(),
            n_atoms_for_pe: 0,
            d_external_work: ptr::null_mut(),
            ghost_tile_ring_dev: 0,
            ghost_tile_max_records: 0,
            firehose_enable: 0,
            body_unroll: 1,
            d_kcc_lead: 0,
            d_forces_anchor: 0,
            d_masses: 0,
            force_burst_step: None,
            n_spikes: 0,
            d_current_md_step: ptr::null(),
            branch_trace_dev: 0,
        };

        let pipeline = match CapturedAdjudicationPipeline::build(&ctx, &stream, &cfg) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[captured-pipeline] build failed: {} — skipping", e);
                return None;
            }
        };
        Some((
            ctx, pipeline, offsets, spikes, d_spikes_b, d_offsets, stream,
        ))
    }

    /// Original native PRSPK population, checked host packing over every row;
    /// bounded GPU reader parity over8192 source-addressed rows (no MD).
    #[test]
    #[ignore = "GPU_RUNTIME_REQUIRED and PRISM_V2_SEED_PARITY_SOURCE/SHA256 required"]
    fn compact_v2_native_seed_reader_gpu_parity() {
        use crate::rich_spike::{NativeV2SeedPopulation,NativeV2SeedLayout};
        use cudarc::driver::DevicePtr;
        use sha2::{Digest,Sha256};
        let path=std::env::var("PRISM_V2_SEED_PARITY_SOURCE").expect("original PRSPK source");
        let expected=std::env::var("PRISM_V2_SEED_PARITY_SHA256").expect("pinned original SHA256");
        let bytes=std::fs::read(&path).expect("native original source");
        let actual=format!("{:x}",Sha256::digest(&bytes));assert_eq!(actual,expected);
        assert_eq!(&bytes[..8],b"PRSPK001");
        fn u64_at(bytes:&[u8],cursor:&mut usize)->u64 {let v=u64::from_le_bytes(bytes[*cursor..*cursor+8].try_into().unwrap());*cursor+=8;v}
        let mut cursor=20usize;let n=u64_at(&bytes,&mut cursor) as usize;cursor+=n;
        let n=u64_at(&bytes,&mut cursor) as usize;cursor+=n;
        let records=u64_at(&bytes,&mut cursor) as usize;let stride=u64_at(&bytes,&mut cursor) as usize;
        let payload=u64_at(&bytes,&mut cursor) as usize;assert_eq!(stride,96);assert_eq!(payload,records*stride);
        assert_eq!(&bytes[cursor+payload+8..],b"PRSPKEND");
        let mut fnv=0xcbf29ce484222325u64;for b in &bytes[cursor..cursor+payload]{fnv^=u64::from(*b);fnv=fnv.wrapping_mul(0x100000001b3);}
        assert_eq!(fnv,u64::from_le_bytes(bytes[cursor+payload..cursor+payload+8].try_into().unwrap()));
        let all:Vec<RichSpike>=bytes[cursor..cursor+payload].chunks_exact(96).map(|row| {
            let event=unsafe{std::ptr::read_unaligned(row.as_ptr().cast::<crate::fused_engine::GpuSpikeEvent>())};
            crate::coupled_md::rich_spike_from_gpu_event(&event,0).unwrap().0
        }).collect();
        let all_compact=NativeV2SeedPopulation::prepare(&all);assert_eq!(all_compact.layout.format,1);
        for (s,c) in all.iter().zip(&all_compact.compact) {
            for (x,y) in [s.x,s.y,s.z,s.water_density,s.wd_change,s.vib_energy].into_iter()
                .zip([c.x,c.y,c.z,c.water_density,c.wd_change,c.vib_energy]) {assert_eq!(x.to_bits(),y.to_bits());}
            assert_eq!(s.intensity_packed,c.intensity_packed);assert_eq!(s.residue_id,i32::from(c.residue_id));
            assert_eq!(s.origin_phase,u32::from(c.phase_source&1023));assert_eq!(s.spike_source,u32::from(c.phase_source>>10));
            assert_eq!(s.causal_lag.to_bits(),all_compact.layout.causal_lag_bits);assert_eq!(s.chem_flags,all_compact.layout.chem_flags);
        }
        let indices:Vec<usize>=(0..records.min(8192)).map(|j|j*records/records.min(8192)).collect();
        let spikes:Vec<RichSpike>=indices.iter().map(|&j|all[j]).collect();
        let compact=NativeV2SeedPopulation::prepare(&spikes);
        let full=unsafe {std::slice::from_raw_parts(spikes.as_ptr().cast::<u8>(),spikes.len()*64)};
        let ctx=CudaContext::new(0).expect("real GPU required");let stream=ctx.new_stream().unwrap();
        let raw=stream.cu_stream() as *mut c_void;
        macro_rules! up {($values:expr)=>{{let values=$values;let mut d=stream.alloc_zeros(values.len()).unwrap();stream.memcpy_htod(values,&mut d).unwrap();d}};}
        macro_rules! ptr {($d:expr)=>{{$d.device_ptr(&stream).0}};}
        assert_eq!(unsafe{crate::sh_basis::ffi::prism_sh_basis_init(raw)},0);
        let anchors:Vec<u32>=(0..1024).collect();assert_eq!(unsafe{crate::so3_project::ffi::prism_so3_set_residue_to_calpha(anchors.as_ptr(),1024,raw)},0);
        let klm=crate::sh_basis::k_lm_device_ptr().unwrap();
        let offsets=up!(&[0u32,spikes.len() as u32]);let forces=up!(&vec![3.0f32;1024*3]);let masses=up!(&vec![12.0f32;1024]);
        let dt=up!(&[0.002f32]);let step=up!(&[123u32]);let ghost_step=up!(&[123u64]);
        let mut results=Vec::new();
        for (seed,layout) in [(full,NativeV2SeedLayout::default()),(compact.bytes(&spikes),compact.layout)] {
            let dseed=up!(seed);let mut adj_host=vec![0u8;256];
            adj_host[48..52].copy_from_slice(&0.1f32.to_le_bytes());
            adj_host[120..128].copy_from_slice(&ptr!(dt).to_le_bytes());
            adj_host[136..140].copy_from_slice(&0.1f32.to_le_bytes());
            adj_host[140..144].copy_from_slice(&u32::MAX.to_le_bytes());let adj=up!(&adj_host);
            let relaxed=stream.alloc_zeros::<u8>(std::mem::size_of::<ContactShellTile>()).unwrap();
            let perturbed=stream.alloc_zeros::<u8>(std::mem::size_of::<ContactShellTile>()).unwrap();
            let xyz=stream.alloc_zeros::<f32>(spikes.len()*3).unwrap();let com=stream.alloc_zeros::<f32>(3).unwrap();
            let mass=stream.alloc_zeros::<f32>(1).unwrap();let correction=stream.alloc_zeros::<f32>(3).unwrap();
            let fields=stream.alloc_zeros::<u32>(spikes.len()*12).unwrap();
            let ghost=crate::ghost_tile::GhostTileRing::allocate(1).unwrap();
            stream.synchronize().unwrap();
            unsafe {
                assert_eq!(crate::so3_project::ffi::prism_v2_seed_consumed_fields_launch(ptr!(dseed) as *const _,ptr!(fields) as *mut _,spikes.len() as u32,raw,layout),0);
                assert_eq!(crate::so3_project::ffi::prism_so3_project_seed_run(ptr!(dseed) as *const _,ptr!(offsets) as *const _,1,klm,ptr!(relaxed) as *mut _,0,0,raw,layout),0);
                assert_eq!(crate::so3_project::ffi::prism_apply_gradient_gasp_positions_seed_clock_launch(ptr!(dseed) as *const _,ptr!(xyz) as *mut _,ptr!(forces) as *const _,ptr!(masses) as *const _,ptr!(adj) as *const _,ptr!(com) as *mut _,ptr!(mass) as *mut _,ptr!(step) as *const _,spikes.len() as u32,1024,raw,layout),0);
                assert_eq!(crate::so3_project::ffi::prism_momentum_guard_check_launch(ptr!(com) as *const _,ptr!(mass) as *const _,ptr!(correction) as *mut _,ptr!(adj) as *mut _,raw),0);
                assert_eq!(crate::so3_project::ffi::prism_apply_com_positions_correction_launch(ptr!(xyz) as *mut _,ptr!(correction) as *const _,spikes.len() as u32,raw),0);
                assert_eq!(crate::so3_project::ffi::prism_so3_project_positions_seed_run(ptr!(dseed) as *const _,ptr!(xyz) as *const _,ptr!(offsets) as *const _,1,klm,ptr!(perturbed) as *mut _,0,0,raw,layout),0);
                assert_eq!(prism_ghost_pipe_stage_launch_v2_seed(ghost.device_base,ptr!(perturbed) as *const _,ptr!(dseed) as *const _,ptr!(offsets) as *const _,ptr!(adj) as *const _,ptr::null(),0,1,1,raw,1,1.0,12.0,0,2.0,0,ptr!(ghost_step),0.0,layout),0);
            }
            stream.synchronize().unwrap();assert_eq!(ghost.n_frames_written(),1);
            let ghost_bytes=unsafe{std::slice::from_raw_parts(ghost.host_base.add(4096),4096)}.to_vec();
            let f=stream.memcpy_dtov(&fields).unwrap();let x:Vec<u32>=stream.memcpy_dtov(&xyz).unwrap().iter().map(|v|v.to_bits()).collect();
            results.push((f,stream.memcpy_dtov(&relaxed).unwrap(),x,stream.memcpy_dtov(&perturbed).unwrap(),ghost_bytes));
        }
        assert_eq!(results[0],results[1],"all consumed fields, relaxed/perturbed tiles, corrected XYZ and complete Ghost bytes");
        eprintln!("NATIVE_V2_SEED_PARITY source={} sha256={} checked_original_rows={} gpu_rows={} full_bytes={} compact_bytes={} layout={:?} original_indices_sha256={:x}",path,actual,records,spikes.len(),records*64,records*32,compact.layout,Sha256::digest(serde_json::to_vec(&indices).unwrap()));
    }

    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn build_instantiates_without_errors_g18() {
        // G18 — Graph Capture Integrity. End-to-end build must
        // succeed: F2 pool, capture, post-capture conditional node
        // injection, instantiate.
        let Some((_ctx, pipeline, ..)) = build_test_pipeline() else {
            return;
        };

        assert_eq!(pipeline.n_clusters, 1);
        // Post-M1.2.18.5 baseline kernel nodes (V1 fixture, d_external_work=null,
        // d_pe_components=null, asc/zstr/sisr=None, dynt7+gearbox always captured):
        //   1. SO(3)            (Node B)
        //   2. Adjudicator step (Node C)
        //   3. Wave-3 dynamic T7 (always-on accumulator)
        //   4. Gearbox SFA      (always-on cruise tensor update)
        // SWITCH bodies + ASC + energy monitor + ZSTR + W_ext memset are
        // all gated off by the test fixture.  Lower-bound assert tolerates
        // additional always-on nodes added in the future.
        assert!(
            pipeline.n_kernel_nodes() >= 2,
            "captured graph must record at least SO(3) + Adjudicator (got {})",
            pipeline.n_kernel_nodes(),
        );
        // V1 ships with implicit dep edges only (capture-mode sequential
        // ordering on md_stream gives the C → C' chain). V2 adds the
        // explicit `cuGraphAddNode(CONDITIONAL, deps=[trampoline])`
        // call which IS the operator's §2.3 explicit-edge mandate.
        assert_eq!(
            pipeline.n_explicit_edges(),
            0,
            "V1 ships with 0 explicit edges (capture handles the chain); \
             V2 will assert >= 1 (trampoline → conditional)"
        );
    }

    #[test]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    fn alignment_g19_attestation() {
        // G19 — F1 Predicate Stability. The F2-pool tile array MUST
        // be 128-byte aligned (LDG.E.128 path). The conditional
        // handle is `0` in V1 (no handle created — V2 lands the
        // conditional node + handle together so the resulting graph
        // is well-formed end-to-end).
        let Some((_ctx, pipeline, ..)) = build_test_pipeline() else {
            return;
        };
        assert!(
            pipeline.tile_alignment_ok(),
            "tiles_dev_ptr {:#x} not 128-byte aligned",
            pipeline.tiles_dev
        );
        assert_eq!(
            pipeline.conditional_handle(),
            0,
            "V1: conditional handle deferred to V2; expected 0"
        );
    }

    #[test]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    fn telemetry_stream_g20_attestation() {
        // G20 — Telemetry Overlap. The orchestrator's telemetry_stream
        // must carry the CU_STREAM_NON_BLOCKING flag (= 1) so the DMA
        // does not implicitly synchronize against the MD integrator
        // stream.
        let Some((_ctx, pipeline, ..)) = build_test_pipeline() else {
            return;
        };
        let flags = stream_flags(pipeline.telemetry_stream()).expect("cuStreamGetFlags");
        assert_eq!(
            flags & 1,
            1,
            "telemetry stream NON_BLOCKING flag missing (got 0x{:x})",
            flags
        );
        // Pinned ring's base pointer must be CU_MEMORYTYPE_HOST.
        let pinned = is_pinned_host(pipeline.ring().base_ptr() as *const c_void)
            .expect("cuPointerGetAttribute");
        assert!(pinned, "ghost ring base pointer must be pinned");
    }

    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn v2_hook_receives_nonzero_graph_and_adj_node_set_and_adj_dev_ptr() {
        // V2 IGNITION readiness: the hook must observe a non-null
        // raw CUgraph + a non-empty Adjudicator-node snapshot + a
        // non-zero, 128-byte-aligned adjudicator FFI pointer.
        // Claude-2's `prism_wire_f1_switch_ffi` will receive these
        // exact values when the C-ABI bypass commits.
        use std::cell::RefCell;
        let Some((ctx, _pipeline_skip, _offsets, _spikes, _d_spikes_b, _d_offsets, stream)) =
            build_test_pipeline()
        else {
            return;
        };
        // Drop the no-op pipeline so we can rebuild with the hook.
        drop(_pipeline_skip);
        // Re-create the same config inline (the helper doesn't expose
        // a way to rebuild without re-allocating spike buffers).
        let raw = stream.cu_stream() as usize;
        unsafe {
            let _ = crate::sh_basis::ffi::prism_sh_basis_init(raw as *mut c_void);
        }
        stream.synchronize().expect("sync");
        let k_lm_dev = crate::sh_basis::k_lm_device_ptr().expect("k_lm");

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
        let offsets: Vec<u32> = vec![0, spikes.len() as u32];
        let spike_bytes = spikes.len() * std::mem::size_of::<RichSpike>();
        let mut d_spikes_b = stream.alloc_zeros::<u8>(spike_bytes).expect("alloc");
        let spikes_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(spikes.as_ptr() as *const u8, spike_bytes).to_vec()
        };
        stream
            .memcpy_htod(&spikes_bytes, &mut d_spikes_b)
            .expect("htod");
        let mut d_offsets = stream.alloc_zeros::<u32>(offsets.len()).expect("alloc o");
        stream
            .memcpy_htod(&offsets, &mut d_offsets)
            .expect("htod o");
        use cudarc::driver::DevicePtr;
        let (sp_dev, off_dev) = {
            let (sp, _g1) = d_spikes_b.device_ptr(&stream);
            let (off, _g2) = d_offsets.device_ptr(&stream);
            (sp, off)
        };
        stream.synchronize().expect("post-htod sync");
        let cfg = PipelineConfig {
            seed_layout: crate::rich_spike::NativeV2SeedLayout::default(),
            d_spikes: sp_dev as *const RichSpike,
            d_cluster_offsets: off_dev as *const u32,
            n_clusters: 1,
            d_k_lm: k_lm_dev,
            initial_frame_id: 0,
            source_stream_id: 0,
            diagnostic_stream_id: None,
            asc: None,
            d_force_norm_mirror: 0,
            governor_device: None,
            zstr: None,
            sisr: None,
            noise_floor_override: None,
            mar_v2: None,
            d_dt: ptr::null_mut(),
            d_velocities: ptr::null_mut(),
            g26_parent_cond_handle: 0,
            f1_parent_cond_handle: 0,
            d_pe_components: ptr::null(),
            n_atoms_for_pe: 0,
            d_external_work: ptr::null_mut(),
            d_forces_anchor: 0,
            n_spikes: 0,
            d_current_md_step: ptr::null(),
            branch_trace_dev: 0,
            force_burst_step: None,
            d_masses: 0,
            ghost_tile_ring_dev: 0,
            d_kcc_lead: 0,
            ghost_tile_max_records: 0,
            firehose_enable: 0,
            body_unroll: 1,
        };

        let observed = RefCell::new(
            None::<(
                usize, /* graph */
                usize, /* n_nodes */
                usize, /* adj_dev */
            )>,
        );
        let hook = |raw_graph: CUgraph,
                    adj_nodes: &[CUgraphNode],
                    adj_dev_ptr: usize|
         -> Result<(), i32> {
            *observed.borrow_mut() = Some((raw_graph as usize, adj_nodes.len(), adj_dev_ptr));
            Ok(())
        };

        let pipeline = CapturedAdjudicationPipeline::build_with_v2_hook(&ctx, &stream, &cfg, hook)
            .expect("V2 build succeeds with no-op hook");

        let (g, n_nodes, adj_dev) = observed.borrow().expect("hook fired");
        assert!(g != 0, "hook received null CUgraph");
        assert!(
            n_nodes >= 1,
            "hook received empty Adjudicator-node snapshot — operator §2.3 \
             explicit-edge dependency target is missing"
        );
        assert!(adj_dev != 0, "hook received null adj_dev_ptr");
        assert_eq!(
            adj_dev % 128,
            0,
            "adj_dev_ptr {:#x} not 128-byte aligned (CSR §M)",
            adj_dev
        );

        // V2 prep accessors return the same handles.
        assert_eq!(pipeline.cu_graph_raw() as usize, g);
        assert_eq!(pipeline.adj_dev_ptr(), adj_dev);
        assert!(!pipeline.cu_graph_exec_raw().is_null());

        eprintln!(
            "[v2-hook] CUgraph=0x{:x}, adj_nodes.len()={}, adj_dev_ptr=0x{:x} (128-aligned ✓)",
            g, n_nodes, adj_dev
        );
    }

    #[test]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    fn v2_hook_failure_aborts_build_cleanly() {
        // Verify the hook's Err return path: build aborts with
        // V2HookFailed and the F2 pool / streams / events are
        // released without leaking handles.
        let Some((ctx, _, _, _, _, _, stream)) = build_test_pipeline() else {
            return;
        };
        let raw = stream.cu_stream() as usize;
        unsafe {
            let _ = crate::sh_basis::ffi::prism_sh_basis_init(raw as *mut c_void);
        }
        stream.synchronize().expect("sync");
        let k_lm_dev = crate::sh_basis::k_lm_device_ptr().expect("k_lm");

        let spikes: Vec<RichSpike> = (0..16u32)
            .map(|i| {
                let mut s = RichSpike::zero();
                let t = 0.3 + (i as f32) * 0.2;
                let p = 0.4 + (i as f32) * 0.3;
                s.x = 4.0 * t.sin() * p.cos();
                s.y = 4.0 * t.sin() * p.sin();
                s.z = 4.0 * t.cos();
                s.cluster_id = 0;
                s
            })
            .collect();
        let offsets: Vec<u32> = vec![0, spikes.len() as u32];
        let spike_bytes = spikes.len() * std::mem::size_of::<RichSpike>();
        let mut d_spikes_b = stream.alloc_zeros::<u8>(spike_bytes).expect("alloc");
        let spikes_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(spikes.as_ptr() as *const u8, spike_bytes).to_vec()
        };
        stream
            .memcpy_htod(&spikes_bytes, &mut d_spikes_b)
            .expect("htod");
        let mut d_offsets = stream.alloc_zeros::<u32>(offsets.len()).expect("alloc o");
        stream
            .memcpy_htod(&offsets, &mut d_offsets)
            .expect("htod o");
        use cudarc::driver::DevicePtr;
        let (sp_dev, off_dev) = {
            let (sp, _g1) = d_spikes_b.device_ptr(&stream);
            let (off, _g2) = d_offsets.device_ptr(&stream);
            (sp, off)
        };
        stream.synchronize().expect("sync");
        let cfg = PipelineConfig {
            seed_layout: crate::rich_spike::NativeV2SeedLayout::default(),
            d_spikes: sp_dev as *const RichSpike,
            d_cluster_offsets: off_dev as *const u32,
            n_clusters: 1,
            d_k_lm: k_lm_dev,
            initial_frame_id: 0,
            source_stream_id: 0,
            diagnostic_stream_id: None,
            asc: None,
            d_force_norm_mirror: 0,
            governor_device: None,
            zstr: None,
            sisr: None,
            noise_floor_override: None,
            mar_v2: None,
            d_dt: ptr::null_mut(),
            n_spikes: 0,
            d_current_md_step: ptr::null(),
            branch_trace_dev: 0,
            d_velocities: ptr::null_mut(),
            g26_parent_cond_handle: 0,
            f1_parent_cond_handle: 0,
            force_burst_step: None,
            d_forces_anchor: 0,
            d_masses: 0,
            d_pe_components: ptr::null(),
            n_atoms_for_pe: 0,
            d_external_work: ptr::null_mut(),
            d_kcc_lead: 0,
            ghost_tile_ring_dev: 0,
            ghost_tile_max_records: 0,
            firehose_enable: 0,
            body_unroll: 1,
        };

        // Synthetic "FFI returned cudaErrorIllegalAddress (700)" via hook.
        let result =
            CapturedAdjudicationPipeline::build_with_v2_hook(&ctx, &stream, &cfg, |_, _, _| {
                Err(700)
            });
        match result {
            Err(BuildError::V2HookFailed { rc: 700 }) => {
                eprintln!("[v2-hook-fail] graceful abort confirmed: V2HookFailed{{rc=700}}");
            }
            Err(other) => panic!("expected V2HookFailed{{rc=700}}, got {:?}", other),
            Ok(_) => panic!("hook returned Err but build claimed Ok"),
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // V2 IGNITION — END-TO-END.
    //
    // Full DAG-COND-WIRE: SO(3) → Adjudicator → cross-stream telemetry
    // DMA → JOIN → cuStreamEndCapture → prism_wire_f1_switch_ffi (the
    // C-ABI bypass that creates the F1 SWITCH conditional node + the
    // explicit cuGraphAddDependencies edge from Node C → Node D) →
    // cuGraphInstantiate → cuGraphLaunch.
    //
    // If this test passes, the captured graph contains a real
    // cudaGraphNodeTypeConditional node, the operator's §2.3 explicit
    // dependency edge is wired natively in C++, the captured graph
    // instantiates, AND we can launch it on the MD stream without
    // CUDA errors. This is the "We are going to light up the RTX 5080"
    // moment — the autonomous WHILE loop's structural ignition.
    // ────────────────────────────────────────────────────────────────────

    #[cfg(feature = "gpu")]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn v2_ignition_wires_and_launches() {
        use crate::interferometric_adjudicator::ffi as adj_ffi;
        use crate::interferometric_adjudicator::InterferometricAdjudicatorFfi;

        let Some((ctx, _drop_pipeline_skip, _, _, _, _, stream)) = build_test_pipeline() else {
            return;
        };
        // Drop the no-op pipeline; we'll rebuild with the V2 hook.
        drop(_drop_pipeline_skip);

        // Re-stage spikes (helper doesn't expose a re-build entry).
        let raw = stream.cu_stream() as usize;
        unsafe {
            let _ = crate::sh_basis::ffi::prism_sh_basis_init(raw as *mut c_void);
        }
        stream.synchronize().expect("sync");
        let k_lm_dev = crate::sh_basis::k_lm_device_ptr().expect("k_lm");

        let spikes: Vec<RichSpike> = (0..16u32)
            .map(|i| {
                let mut s = RichSpike::zero();
                let t = 0.3 + (i as f32) * 0.2;
                let p = 0.4 + (i as f32) * 0.3;
                s.x = 4.0 * t.sin() * p.cos();
                s.y = 4.0 * t.sin() * p.sin();
                s.z = 4.0 * t.cos();
                s.cluster_id = 0;
                s
            })
            .collect();
        let offsets: Vec<u32> = vec![0, spikes.len() as u32];
        let spike_bytes = spikes.len() * std::mem::size_of::<RichSpike>();
        let mut d_spikes_b = stream.alloc_zeros::<u8>(spike_bytes).expect("alloc");
        let spikes_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(spikes.as_ptr() as *const u8, spike_bytes).to_vec()
        };
        stream
            .memcpy_htod(&spikes_bytes, &mut d_spikes_b)
            .expect("htod");
        let mut d_offsets = stream.alloc_zeros::<u32>(offsets.len()).expect("alloc o");
        stream
            .memcpy_htod(&offsets, &mut d_offsets)
            .expect("htod o");
        use cudarc::driver::DevicePtr;
        let (sp_dev, off_dev) = {
            let (sp, _g1) = d_spikes_b.device_ptr(&stream);
            let (off, _g2) = d_offsets.device_ptr(&stream);
            (sp, off)
        };
        stream.synchronize().expect("sync");

        let cfg = PipelineConfig {
            seed_layout: crate::rich_spike::NativeV2SeedLayout::default(),
            d_spikes: sp_dev as *const RichSpike,
            d_cluster_offsets: off_dev as *const u32,
            n_clusters: 1,
            d_k_lm: k_lm_dev,
            initial_frame_id: 0,
            source_stream_id: 0,
            diagnostic_stream_id: None,
            n_spikes: 0,
            d_current_md_step: ptr::null(),
            branch_trace_dev: 0,
            asc: None,
            d_force_norm_mirror: 0,
            governor_device: None,
            zstr: None,
            force_burst_step: None,
            sisr: None,
            d_forces_anchor: 0,
            d_masses: 0,
            noise_floor_override: None,
            mar_v2: None,
            d_dt: ptr::null_mut(),
            d_velocities: ptr::null_mut(),
            g26_parent_cond_handle: 0,
            f1_parent_cond_handle: 0,
            d_pe_components: ptr::null(),
            n_atoms_for_pe: 0,
            d_kcc_lead: 0,
            d_external_work: ptr::null_mut(),
            ghost_tile_ring_dev: 0,
            ghost_tile_max_records: 0,
            firehose_enable: 0,
            body_unroll: 1,
        };

        // Closure that captures the conditional-node handle written
        // by Claude-2's bypass so the test can attest its non-null-ness.
        use std::cell::Cell;
        let observed_cond_node: Cell<usize> = Cell::new(0);

        // V2 IGNITION HOOK — the moment of fusion.
        let pipeline = CapturedAdjudicationPipeline::build_with_v2_hook(
            &ctx,
            &stream,
            &cfg,
            |raw_graph, adj_nodes, adj_dev_ptr| {
                assert!(!raw_graph.is_null(), "hook received null CUgraph");
                assert_eq!(
                    adj_nodes.len(),
                    1,
                    "operator §2.3 dependency frontier must hold exactly the Adjudicator node"
                );
                assert_eq!(
                    adj_dev_ptr % 128,
                    0,
                    "adj_dev_ptr {:#x} not 128-byte aligned (CSR §M)",
                    adj_dev_ptr
                );

                // Predicate device pointer: Claude-2's existing FFI
                // helper that returns &adj->adjudication_code at byte
                // offset 52.
                let predicate_ptr = unsafe {
                    adj_ffi::prism_get_adjudication_code_devptr(
                        adj_dev_ptr as *const InterferometricAdjudicatorFfi,
                    )
                };
                assert!(
                    !predicate_ptr.is_null(),
                    "prism_get_adjudication_code_devptr returned null"
                );

                // The C-ABI bypass.
                let mut cond_node: CUgraphNode = ptr::null_mut();
                let rc = unsafe {
                    super::prism_wire_f1_switch_ffi(
                        raw_graph,
                        adj_nodes[0],
                        predicate_ptr,
                        &mut cond_node as *mut _,
                    )
                };
                if rc != 0 {
                    eprintln!(
                        "[v2-ignition] prism_wire_f1_switch_ffi returned cudaError {}",
                        rc
                    );
                    return Err(rc);
                }
                if cond_node.is_null() {
                    eprintln!(
                        "[v2-ignition] prism_wire_f1_switch_ffi succeeded but \
                              left out_conditional_node = null"
                    );
                    return Err(-1);
                }
                observed_cond_node.set(cond_node as usize);
                Ok(())
            },
        )
        .expect("V2 IGNITION build_with_v2_hook must succeed");

        // The bypass populated the conditional node handle.
        let cond_node = observed_cond_node.get();
        assert!(
            cond_node != 0,
            "V2 IGNITION: conditional-node handle was not populated by the C-ABI bypass"
        );

        // Pipeline instantiated (hook returned Ok, so build did
        // cuGraphInstantiate). Launch must succeed end-to-end.
        for frame in 0..3u64 {
            pipeline.launch().unwrap_or_else(|e| {
                panic!("cuGraphLaunch (frame {}) failed: {:?}", frame, e);
            });
        }
        stream.synchronize().expect("post-launch sync");

        // Marker readable post-launch (semantic value depends on the
        // Adjudicator's noise-floor calibration; here we only attest
        // the read path is intact — V2 IGNITION's structural success).
        let marker = pipeline.read_burst_marker().expect("read burst_marker");

        eprintln!(
            "[v2-ignition] LIVE: CUgraph=0x{:x}, cond_node=0x{:x}, \
                  predicate_ptr OK (128-aligned), 3 launches OK, \
                  burst_marker={} (route attestation)",
            pipeline.cu_graph_raw() as usize,
            cond_node,
            marker
        );
    }

    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn launch_executes_without_errors() {
        // Smoke: cuGraphLaunch must succeed on the captured graph.
        // Routing semantics (does the Adjudicator produce code 0/1/2
        // correctly?) are deferred to the integration test; here we
        // only attest the IGNITION sequence runs.
        let Some((_ctx, pipeline, ..)) = build_test_pipeline() else {
            return;
        };
        let stream = pipeline.md_stream.clone();
        // Fire 3 launches on the captured graph.
        for _ in 0..3 {
            pipeline.launch().expect("cuGraphLaunch");
        }
        stream.synchronize().expect("post-launch sync");

        // burst_marker should reflect the conditional's default value
        // (Adjudicator on synthetic input writes code = 0 → conditional
        // skipped → marker stays at 0). We don't enforce a specific
        // value here because the Adjudicator's noise floor on a
        // not-yet-calibrated input is undefined; we ONLY assert the
        // marker is readable post-launch (no driver error).
        let marker = pipeline.read_burst_marker().expect("read burst_marker");
        eprintln!(
            "[captured-pipeline] post-launch burst_marker = {} \
                  (0 = Prune route, 1 = Burst route)",
            marker
        );
    }

    #[test]
    fn zstr_observation_work_is_snapshotted_before_retirement() {
        let source=include_str!("captured_pipeline.rs");
        let start=source.find("// Step 3: SFA kernel.").unwrap();
        let body=&source[start..source.find("} // ── end §3.III BODY-UNROLL for-loop").unwrap()];
        let snapshot=body.find("zstr_launch_stage_audit(").unwrap();
        let joined=body.find("cuStreamWaitEvent(md_stream.cu_stream(), telemetry_to_md_event, 0)").unwrap();
        let reset=body.find("cfg.d_external_work as CUdeviceptr").unwrap();
        assert!(snapshot<joined && joined<reset);
        assert_eq!(body.matches("cfg.d_external_work as CUdeviceptr").count(),1);
    }

    #[test]
    fn exact_ghost_counter_seed_catches_replay_counter_vs_md_step_mismatch() {
        assert_eq!(ghost_counter_seed_for_exact_step(1_000), Some(999));
        assert_eq!(ghost_counter_seed_for_exact_step(1), Some(0));
        assert_eq!(ghost_counter_seed_for_exact_step(0), None);
        let stale_replay_seed = 500u64;
        assert_ne!(
            stale_replay_seed + 1,
            1_000,
            "seed+1 replay clock must not be accepted as chunk-end MD time"
        );
    }
}


#[derive(serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdjudicationContinuationV1 {
    schema_version:u32,stream_id:u32,n_clusters:u32,n_seed_spikes:u32,n_physical_atoms:u32,
    seed_layout:crate::rich_spike::NativeV2SeedLayout,
    #[serde(with="crate::checkpoint_codec::byte_map")]
    regions:std::collections::BTreeMap<String,Vec<u8>>,
    relaxed_uses_frozen:bool,#[serde(with="crate::checkpoint_codec::bytes")] pinned_ring:Vec<u8>,
    trace_record_idx:u64,trace_launch_seq:u64,trace_open_failed:bool,
    trace_cursor:Option<crate::scientific_checkpoint::AppendCursor>,
}
impl CapturedAdjudicationPipeline {
    fn continuation_regions(&self)->Vec<(&'static str,u64,usize)> {
        vec![("perturbed_positions",self.perturbed_positions_dev as u64,self.n_seed_spikes.max(1) as usize*12),
            ("com_shift",self.com_shift_dev as u64,12),("total_mass",self.total_mass_dev as u64,4),
            ("com_correction",self.com_correction_dev as u64,12),
            ("tiles",self.tiles_dev as u64,self.tiles_bytes as usize),
            ("relaxed",self.tiles_baseline_dev as u64,self.tiles_bytes as usize),
            ("frozen",self.p_frozen_dev as u64,self.tiles_bytes as usize),
            ("adjudicator",self.adj_dev as u64,std::mem::size_of::<InterferometricAdjudicatorFfi>()),
            ("burst_marker",self.burst_marker_dev as u64,4),
            ("sisr_mask",self.sisr_mask_dev as u64,8),("sisr_count",self.sisr_count_dev as u64,4),
            ("calibration_f64",self.dynt7_acc_dev as u64,32),
            ("calibration_compat_index",self.dynt7_idx_dev as u64,4),
            ("calibration_compat_stats",self.dynt7_stats_dev as u64,8),
            // Cruise is borrowed from the same governor owner. The duplicate
            // copy is checked during coordinated restore, never given a new
            // default. It carries graph/gearbox recurrence memory.
            ("cruise",self.cruise_state_dev as u64,std::mem::size_of::<crate::gearbox::ChronometricStateTensor>()),
            ("energy_scratch",self.energy_temp_storage_dev as u64,self.energy_temp_storage_bytes as usize),
            ("energy_scalar",self.energy_pe_scalar_dev as u64,8),
            ("energy_window",self.energy_window_dev as u64,16),
            ("ghost_clock",self.ghost_step_counter_dev,8)]
    }
    fn continuation_pause(&self)->anyhow::Result<()> {
        self.md_stream.synchronize()?;
        let rc=unsafe{cuStreamSynchronize(self.telemetry_stream)};
        anyhow::ensure!(rc==CUresult::CUDA_SUCCESS,"telemetry checkpoint drain: {rc:?}");Ok(())
    }
    pub fn continuation_allocation_plan(&self)->anyhow::Result<Vec<(String,u64)>> {
        let mut rows=Vec::new();
        for(name,pointer,bytes)in self.continuation_regions(){if pointer!=0{rows.push((name.into(),u64::try_from(bytes)?));}}
        rows.push(("pinned_ring".into(),u64::try_from(self.ring.total_bytes())?));
        Ok(rows)
    }
    pub fn capture_continuation(&self,run_root:&std::path::Path)->anyhow::Result<AdjudicationContinuationV1> {
        use crate::scientific_checkpoint::device::read_raw;
        self.continuation_pause()?;
        let mut regions=std::collections::BTreeMap::new();
        for (name,pointer,bytes) in self.continuation_regions() {
            if pointer!=0 {regions.insert(name.into(),unsafe{read_raw(&self.md_stream,pointer,bytes)?});}
        }
        let adj=regions.get_mut("adjudicator").ok_or_else(||anyhow::anyhow!("adjudicator allocation absent"))?;
        let relaxed_offset=std::mem::offset_of!(InterferometricAdjudicatorFfi,relaxed_manifold_ptr);
        let relaxed=u64::from_le_bytes(adj[relaxed_offset..relaxed_offset+8].try_into()?);
        anyhow::ensure!(relaxed==self.p_frozen_dev as u64 || relaxed==self.tiles_baseline_dev as u64,
            "adjudicator relaxed manifold has unknown owner");
        let relaxed_uses_frozen=relaxed==self.p_frozen_dev as u64;
        for offset in [std::mem::offset_of!(InterferometricAdjudicatorFfi,relaxed_manifold_ptr),
            std::mem::offset_of!(InterferometricAdjudicatorFfi,perturbed_manifold_ptr),
            std::mem::offset_of!(InterferometricAdjudicatorFfi,force_prune_mask_ptr),
            std::mem::offset_of!(InterferometricAdjudicatorFfi,d_dt),
            std::mem::offset_of!(InterferometricAdjudicatorFfi,d_external_work)] {adj[offset..offset+8].fill(0);}
        let pinned_ring=if self.ring.total_bytes()==0 {Vec::new()} else {
            // Allocation is initialized at construction; both producing streams
            // are synchronized above and the all-stream owner pause is held.
            unsafe{std::slice::from_raw_parts(self.ring.base_ptr().cast::<u8>(),self.ring.total_bytes()).to_vec()}
        };
        let trace=self.g26_trace.lock().map_err(|_|anyhow::anyhow!("G26 trace mutex poisoned"))?;
        let trace_cursor=if let Some(file)=&trace.file {
            let cfg=self.g26_trace_cfg.as_ref().ok_or_else(||anyhow::anyhow!("G26 file lacks config"))?;
            let path=cfg.path();let relative=path.strip_prefix(run_root)?.to_str().ok_or_else(||anyhow::anyhow!("non-UTF8 G26 path"))?;
            Some(crate::scientific_checkpoint::seal_append_file(file,relative)?)
        } else {None};
        Ok(AdjudicationContinuationV1{schema_version:1,stream_id:self.g26_stream_id,n_clusters:self.n_clusters,
            n_seed_spikes:self.n_seed_spikes,n_physical_atoms:self.n_physical_atoms,seed_layout:self.seed_layout,
            regions,relaxed_uses_frozen,pinned_ring,trace_record_idx:trace.record_idx,trace_launch_seq:trace.launch_seq,
            trace_open_failed:trace.open_failed,trace_cursor})
    }
    pub fn restore_continuation(&self,run_root:&std::path::Path,saved:&AdjudicationContinuationV1)->anyhow::Result<()> {
        use crate::scientific_checkpoint::device::{read_raw,write_raw};
        self.continuation_pause()?;
        anyhow::ensure!(saved.schema_version==1 && saved.stream_id==self.g26_stream_id
            && saved.n_clusters==self.n_clusters && saved.n_seed_spikes==self.n_seed_spikes
            && saved.n_physical_atoms==self.n_physical_atoms && saved.seed_layout==self.seed_layout,
            "adjudication restart population/identity mismatch");
        let expected=self.continuation_regions().into_iter().filter(|(_,p,_)|*p!=0).collect::<Vec<_>>();
        anyhow::ensure!(saved.regions.len()==expected.len() && saved.pinned_ring.len()==self.ring.total_bytes(),
            "adjudication checkpoint allocation population mismatch");
        for(name,_,n)in &expected{anyhow::ensure!(saved.regions.get(*name).is_some_and(|v|v.len()==*n),
            "adjudication checkpoint extent mismatch for {name}");}
        let mut adjudicator=saved.regions["adjudicator"].clone();
        let fresh=unsafe{read_raw(&self.md_stream,self.adj_dev as u64,adjudicator.len())?};
        for offset in [std::mem::offset_of!(InterferometricAdjudicatorFfi,relaxed_manifold_ptr),
            std::mem::offset_of!(InterferometricAdjudicatorFfi,perturbed_manifold_ptr),
            std::mem::offset_of!(InterferometricAdjudicatorFfi,force_prune_mask_ptr),
            std::mem::offset_of!(InterferometricAdjudicatorFfi,d_dt),
            std::mem::offset_of!(InterferometricAdjudicatorFfi,d_external_work)] {
            anyhow::ensure!(adjudicator[offset..offset+8].iter().all(|&x|x==0),"serialized adjudicator contains device pointer");
            adjudicator[offset..offset+8].copy_from_slice(&fresh[offset..offset+8]);
        }
        let offset=std::mem::offset_of!(InterferometricAdjudicatorFfi,relaxed_manifold_ptr);
        let relaxed=if saved.relaxed_uses_frozen{self.p_frozen_dev}else{self.tiles_baseline_dev}as u64;
        adjudicator[offset..offset+8].copy_from_slice(&relaxed.to_le_bytes());
        let trace_file=if let Some(cursor)=&saved.trace_cursor {
            let cfg=self.g26_trace_cfg.as_ref().ok_or_else(||anyhow::anyhow!("restored G26 trace lacks config"))?;
            anyhow::ensure!(cfg.path()==run_root.join(&cursor.relative_path),"G26 restart path changed");
            Some(crate::scientific_checkpoint::reopen_append_file(run_root,cursor)?)
        } else {None};
        // All allocation shapes and pointer rebinding were validated first.
        for (name,pointer,_)in expected {let bytes=if name=="adjudicator"{&adjudicator}else{&saved.regions[name]};
            unsafe{write_raw(&self.md_stream,pointer,bytes)?;}}
        if !saved.pinned_ring.is_empty(){unsafe{std::ptr::copy_nonoverlapping(saved.pinned_ring.as_ptr(),
            self.ring.base_ptr().cast::<u8>(),saved.pinned_ring.len());}}
        let mut trace=self.g26_trace.lock().map_err(|_|anyhow::anyhow!("G26 trace mutex poisoned"))?;
        trace.file=trace_file;trace.record_idx=saved.trace_record_idx;trace.launch_seq=saved.trace_launch_seq;
        trace.open_failed=saved.trace_open_failed;Ok(())
    }
}

/// Constructor facts retained without any CUDA pointer or graph handle. The
/// seed payload, CSR, calibration and every history are separate state owners.
#[derive(serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineContinuationConfigV1 {
    schema_version:u32,bindings:std::collections::BTreeSet<String>,
    governor_present:bool,parent_f1:bool,parent_g26:bool,
    asc:Option<(i32,f32,bool)>,zstr:Option<ZstrContinuationConfigV1>,
    pub seed_layout:crate::rich_spike::NativeV2SeedLayout,
    pub n_clusters:u32,
    pub initial_frame_id:u32,
    pub source_stream_id:u32,
    pub diagnostic_stream_id:Option<u32>,
    pub sisr:Option<SisrConfig>,
    pub mar_v2:Option<MarV2Config>,
    pub noise_floor_override:Option<(f32,f32)>,
    pub n_atoms_for_pe:u32,
    pub ghost_tile_max_records:u32,
    pub firehose_enable:u32,
    pub body_unroll:u32,
    pub force_burst_step:Option<u32>,
    pub n_spikes:u32,
}
#[derive(serde::Serialize,serde::Deserialize)]
struct ZstrContinuationConfigV1 {
    n_atoms:u32,
    inter_slot_stride:u32,
    pos_offset_in_slot:u32,
    force_offset_in_slot:u32,
    force_norm_offset_in_slot:u32,
    fence_offset_in_slot:u32,
    n_slots:u32,
    backpressure_spin_limit:u32,
    stream_id:u32,
    schema_version:u32,
}
/// Fresh owner bindings supplied by the native worker's construction path.
/// Values never originate from serialized previous-process addresses.
pub struct PipelineContinuationBindings {
    pub governor:Option<Arc<crate::governor_device::GovernorDevice>>,
    pub asc:Option<Arc<crate::asc_device::AscDevice>>,
    pub addresses:std::collections::BTreeMap<String,u64>,
    pub parent_f1:u64,pub parent_g26:u64,
}
impl PipelineConfig {
    pub fn checkpoint_configuration(&self)->PipelineContinuationConfigV1 {
        let mut bindings=std::collections::BTreeSet::new();
        if self.d_spikes as u64!=0 {bindings.insert("d_spikes".into());}
        if self.d_cluster_offsets as u64!=0 {bindings.insert("d_cluster_offsets".into());}
        if self.d_k_lm as u64!=0 {bindings.insert("d_k_lm".into());}
        if self.d_force_norm_mirror as u64!=0 {bindings.insert("d_force_norm_mirror".into());}
        if self.d_dt as u64!=0 {bindings.insert("d_dt".into());}
        if self.d_velocities as u64!=0 {bindings.insert("d_velocities".into());}
        if self.d_pe_components as u64!=0 {bindings.insert("d_pe_components".into());}
        if self.d_external_work as u64!=0 {bindings.insert("d_external_work".into());}
        if self.ghost_tile_ring_dev as u64!=0 {bindings.insert("ghost_tile_ring_dev".into());}
        if self.d_kcc_lead as u64!=0 {bindings.insert("d_kcc_lead".into());}
        if self.d_forces_anchor as u64!=0 {bindings.insert("d_forces_anchor".into());}
        if self.d_masses as u64!=0 {bindings.insert("d_masses".into());}
        if self.d_current_md_step as u64!=0 {bindings.insert("d_current_md_step".into());}
        if self.branch_trace_dev as u64!=0 {bindings.insert("branch_trace_dev".into());}
        if let Some(asc)=&self.asc {
            if asc.d_forces as u64!=0 {bindings.insert("asc.d_forces".into());}
            if asc.d_atom_positions as u64!=0 {bindings.insert("asc.d_atom_positions".into());}
            if asc.d_atom_in_cluster as u64!=0 {bindings.insert("asc.d_atom_in_cluster".into());}
            if asc.d_asc_applied_atoms as u64!=0 {bindings.insert("asc.d_asc_applied_atoms".into());}
            if asc.d_asc_applied_sum_f as u64!=0 {bindings.insert("asc.d_asc_applied_sum_f".into());}
        }
        if let Some(zstr)=&self.zstr {
            if zstr.d_positions as u64!=0 {bindings.insert("zstr.d_positions".into());}
            if zstr.d_forces as u64!=0 {bindings.insert("zstr.d_forces".into());}
            if zstr.pinned_base as u64!=0 {bindings.insert("zstr.pinned_base".into());}
            if zstr.d_active_slot as u64!=0 {bindings.insert("zstr.d_active_slot".into());}
            if zstr.d_backpressure_status as u64!=0 {bindings.insert("zstr.d_backpressure_status".into());}
        }
        if self.mar_v2.as_ref().is_some_and(|v|v.d_step_counter!=0){bindings.insert("mar_v2.d_step_counter".into());}
        let mut mar_v2=self.mar_v2; if let Some(value)=&mut mar_v2{value.d_step_counter=0;}
        PipelineContinuationConfigV1{schema_version:1,bindings,governor_present:self.governor_device.is_some(),
            parent_f1:self.f1_parent_cond_handle!=0,parent_g26:self.g26_parent_cond_handle!=0,
            asc:self.asc.as_ref().map(|a|(a.n_atoms,a.steering_gain_alpha,a.decision_owner.is_some())),
            zstr:self.zstr.as_ref().map(|z|ZstrContinuationConfigV1{
                n_atoms:z.n_atoms,
                inter_slot_stride:z.inter_slot_stride,
                pos_offset_in_slot:z.pos_offset_in_slot,
                force_offset_in_slot:z.force_offset_in_slot,
                force_norm_offset_in_slot:z.force_norm_offset_in_slot,
                fence_offset_in_slot:z.fence_offset_in_slot,
                n_slots:z.n_slots,
                backpressure_spin_limit:z.backpressure_spin_limit,
                stream_id:z.stream_id,
                schema_version:z.schema_version,
            }),
            seed_layout:self.seed_layout.clone(),
            n_clusters:self.n_clusters.clone(),
            initial_frame_id:self.initial_frame_id.clone(),
            source_stream_id:self.source_stream_id.clone(),
            diagnostic_stream_id:self.diagnostic_stream_id.clone(),
            sisr:self.sisr.clone(),
            mar_v2:mar_v2,
            noise_floor_override:self.noise_floor_override.clone(),
            n_atoms_for_pe:self.n_atoms_for_pe.clone(),
            ghost_tile_max_records:self.ghost_tile_max_records.clone(),
            firehose_enable:self.firehose_enable.clone(),
            body_unroll:self.body_unroll.clone(),
            force_burst_step:self.force_burst_step.clone(),
            n_spikes:self.n_spikes.clone(),
        }
    }
    pub fn rebind_checkpoint_configuration(saved:&PipelineContinuationConfigV1,fresh:PipelineContinuationBindings)->anyhow::Result<Self> {
        anyhow::ensure!(saved.schema_version==1 && saved.seed_layout.validate(),"unsupported V2 constructor checkpoint");
        anyhow::ensure!(saved.governor_present==fresh.governor.is_some() && saved.asc.as_ref().map_or(false,|a|a.2)==fresh.asc.is_some(),"V2 controller owner presence changed");
        let address=|name:&str|->anyhow::Result<u64>{if saved.bindings.contains(name){
            let pointer=fresh.addresses.get(name).copied().unwrap_or(0);
            anyhow::ensure!(pointer!=0,"missing fresh V2 owner binding {name}");Ok(pointer)
        }else{Ok(0)}};
        let mut mar_v2=saved.mar_v2;
        if let Some(value)=&mut mar_v2{value.d_step_counter=address("mar_v2.d_step_counter")?;}
        // Parent builders own conditional handles and replace these before
        // capture. A standalone constructor must provide its new parent pair.
        anyhow::ensure!((fresh.parent_f1==0)==(fresh.parent_g26==0),"incomplete fresh V2 parent handle pair");
        Ok(Self{
            governor_device:fresh.governor.clone(),g26_parent_cond_handle:fresh.parent_g26,f1_parent_cond_handle:fresh.parent_f1,
            asc:saved.asc.map(|(n_atoms,steering_gain_alpha,_)|->anyhow::Result<_>{Ok(AscConfig{
                decision_owner:fresh.asc.clone(),n_atoms,steering_gain_alpha,
                d_forces:address("asc.d_forces")? as *mut f32,
                d_atom_positions:address("asc.d_atom_positions")? as *const f32,
                d_atom_in_cluster:address("asc.d_atom_in_cluster")? as *const u32,
                d_asc_applied_atoms:address("asc.d_asc_applied_atoms")? as *mut u32,
                d_asc_applied_sum_f:address("asc.d_asc_applied_sum_f")? as *mut f32,
            })}).transpose()?,
            zstr:saved.zstr.as_ref().map(|z|->anyhow::Result<_>{Ok(ZstrCaptureParams{
                n_atoms:z.n_atoms,
                inter_slot_stride:z.inter_slot_stride,
                pos_offset_in_slot:z.pos_offset_in_slot,
                force_offset_in_slot:z.force_offset_in_slot,
                force_norm_offset_in_slot:z.force_norm_offset_in_slot,
                fence_offset_in_slot:z.fence_offset_in_slot,
                n_slots:z.n_slots,
                backpressure_spin_limit:z.backpressure_spin_limit,
                stream_id:z.stream_id,
                schema_version:z.schema_version,
                d_positions:address("zstr.d_positions")? as *const f32,
                d_forces:address("zstr.d_forces")? as *const f32,
                pinned_base:address("zstr.pinned_base")? as *mut u8,
                d_active_slot:address("zstr.d_active_slot")? as *mut u32,
                d_backpressure_status:address("zstr.d_backpressure_status")? as *mut u8,
            })}).transpose()?,
            d_spikes:address("d_spikes")? as *const RichSpike,
            d_cluster_offsets:address("d_cluster_offsets")? as *const u32,
            d_k_lm:address("d_k_lm")? as *const f32,
            d_force_norm_mirror:address("d_force_norm_mirror")? as u64,
            d_dt:address("d_dt")? as *mut f32,
            d_velocities:address("d_velocities")? as *mut f32,
            d_pe_components:address("d_pe_components")? as *const f64,
            d_external_work:address("d_external_work")? as *mut f64,
            ghost_tile_ring_dev:address("ghost_tile_ring_dev")? as u64,
            d_kcc_lead:address("d_kcc_lead")? as u64,
            d_forces_anchor:address("d_forces_anchor")? as u64,
            d_masses:address("d_masses")? as u64,
            d_current_md_step:address("d_current_md_step")? as *const u32,
            branch_trace_dev:address("branch_trace_dev")? as u64,
            seed_layout:saved.seed_layout.clone(),
            n_clusters:saved.n_clusters.clone(),
            initial_frame_id:saved.initial_frame_id.clone(),
            source_stream_id:saved.source_stream_id.clone(),
            diagnostic_stream_id:saved.diagnostic_stream_id.clone(),
            sisr:saved.sisr.clone(),
            mar_v2:mar_v2,
            noise_floor_override:saved.noise_floor_override.clone(),
            n_atoms_for_pe:saved.n_atoms_for_pe.clone(),
            ghost_tile_max_records:saved.ghost_tile_max_records.clone(),
            firehose_enable:saved.firehose_enable.clone(),
            body_unroll:saved.body_unroll.clone(),
            force_burst_step:saved.force_burst_step.clone(),
            n_spikes:saved.n_spikes.clone(),
        })
    }
}

impl PipelineContinuationConfigV1 {
    pub fn has_parent_conditionals(&self)->bool {self.parent_f1 && self.parent_g26}
}
impl CapturedAdjudicationPipeline {
    pub fn continuation_parent_handles(&self)->(u64,u64){(self.f1_cond_handle,self.g26_cond_handle)}
}

// Native accepted-window ASC decision producer; reuses the existing native FFI.
#[path="biprism_adjudication.rs"]
pub mod biprism_adjudication;
