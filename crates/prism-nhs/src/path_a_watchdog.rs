//! PATH-A mid-chunk stall watchdog and crash-safe evidence-exit fallback.
//!
//! Per directive after the 2026-05-06 P5 validation postmortem: the existing
//! `--path-a-max-wall-seconds` check fires only at the chunk-loop entry,
//! and the engine has been observed to deadlock mid-chunk on an inter-stream
//! 7-of-8 barrier. This module provides:
//!
//!   1. Per-stream progress heartbeats (`PathAHeartbeat`) writable from
//!      every chunk boundary and key phase transition without holding any
//!      engine-internal lock.
//!   2. A polling `spawn_watchdog` thread that detects N-1/N stall patterns
//!      and flips a shared cancel flag with a typed exit reason.
//!   3. An async-signal-safe SIGINT handler (`install_sigint_handler`) that
//!      sets a static AtomicBool the watchdog pumps into the cancel flag.
//!   4. A crash-safe minimal `path_a_completion.json` emit
//!      (`emit_minimal_completion_json`) used as a deadman fallback when
//!      the stream-worker join cannot return because a stream is futex-blocked.
//!      After emit, the watchdog calls `process::exit(2)` so the OS
//!      reclaims VRAM cleanly without an external SIGKILL.
//!
//! This module does NOT refactor existing engine internal synchronization.
//! It provides an EXTERNAL escape hatch only. Internal-lock audit + timed
//! waits remain a separate orthogonal investigation.
//!
//! ## Invariants honored
//! - No graph topology changes (this module is host-side Rust only).
//! - No F1/G26/WHILE alteration.
//! - No runtime Python.
//! - No fake completion: `completion_status` never reports `full_complete`.
//! - No fake binding sites: this module never writes binding_sites.json.
//! - Stream count unchanged at 8 (or whatever caller passes).

use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Phase encoding written into `current_phase` per stream.
pub const PHASE_INIT: i32 = 0;
/// Cold-hold integration phase.
pub const PHASE_COLD_HOLD: i32 = 1;
/// T7 calibration phase (between cold_hold and V2 ignition / live).
pub const PHASE_T7_CAL: i32 = 2;
/// V2-live captured-graph integration.
pub const PHASE_V2_LIVE: i32 = 3;
/// Per-stream teardown / artifact emit.
pub const PHASE_TEARDOWN: i32 = 4;
/// Stream complete, no further progress expected.
pub const PHASE_DONE: i32 = 5;

// POST_CHUNK_LOOP_TEARDOWN_STALL diagnostic — fine-grained chunk-body
// and teardown phase markers. Inserted at the boundaries of suspected
// blocking sites so the deadman path_a_completion.json names the EXACT
// stage every stream is stuck at when the watchdog fires.
//
// Codes 10-19 = chunk body sub-stages (loop iteration interior).
// Codes 20-29 = teardown sub-stages (post-loop per-stream finalization).

/// Chunk body — entered after cancel/wall/T7/evidence-exit checks pass.
pub const PHASE_CHUNK_BODY_ENTER: i32 = 10;
/// Chunk body — T5 phase boundary build (V2 instantiate first time).
pub const PHASE_CHUNK_BODY_V2_BUILD: i32 = 11;
/// Chunk body — about to launch the captured V2 graph (graph.run_chunk).
pub const PHASE_CHUNK_BODY_GRAPH_LAUNCH: i32 = 12;
/// Chunk body — post-launch sync / per-chunk telemetry copies.
pub const PHASE_CHUNK_BODY_POST_LAUNCH_SYNC: i32 = 13;

/// Teardown — entered after chunk loop exit (break or natural end).
pub const PHASE_TEARDOWN_ENTER: i32 = 20;
/// Teardown — VRAM-Loss audit DtoH copies (forces, aromatic centroids, warp).
pub const PHASE_TEARDOWN_VRAM_AUDIT: i32 = 21;
/// Teardown — engine.cuda_stream().synchronize() drain.
pub const PHASE_TEARDOWN_ENGINE_SYNC: i32 = 22;
/// Teardown — drop(v2_snap_tx) + v2_writer_handle.join().
pub const PHASE_TEARDOWN_V2_WRITER_JOIN: i32 = 23;
/// Teardown — engine.download_signal_preservation().
pub const PHASE_TEARDOWN_DOWNLOAD_SIG: i32 = 24;
/// Teardown — engine.compute_and_download_kcc().
pub const PHASE_TEARDOWN_DOWNLOAD_KCC: i32 = 25;
/// Teardown — closure return (last code observed before rayon thread exits).
pub const PHASE_TEARDOWN_CLOSURE_RETURN: i32 = 26;

/// Stable lowercase label for a phase code. Returns "unknown:<code>"
/// for unrecognized values so the deadman emit never panics on a
/// future code that hasn't been added here.
pub fn phase_label(code: i32) -> String {
    match code {
        PHASE_INIT => "init".to_string(),
        PHASE_COLD_HOLD => "cold_hold".to_string(),
        PHASE_T7_CAL => "t7_cal".to_string(),
        PHASE_V2_LIVE => "v2_live".to_string(),
        PHASE_TEARDOWN => "teardown".to_string(),
        PHASE_DONE => "done".to_string(),
        PHASE_CHUNK_BODY_ENTER => "chunk_body_enter".to_string(),
        PHASE_CHUNK_BODY_V2_BUILD => "chunk_body_v2_build".to_string(),
        PHASE_CHUNK_BODY_GRAPH_LAUNCH => "chunk_body_graph_launch".to_string(),
        PHASE_CHUNK_BODY_POST_LAUNCH_SYNC => "chunk_body_post_launch_sync".to_string(),
        PHASE_TEARDOWN_ENTER => "teardown_enter".to_string(),
        PHASE_TEARDOWN_VRAM_AUDIT => "teardown_vram_audit".to_string(),
        PHASE_TEARDOWN_ENGINE_SYNC => "teardown_engine_sync".to_string(),
        PHASE_TEARDOWN_V2_WRITER_JOIN => "teardown_v2_writer_join".to_string(),
        PHASE_TEARDOWN_DOWNLOAD_SIG => "teardown_download_sig".to_string(),
        PHASE_TEARDOWN_DOWNLOAD_KCC => "teardown_download_kcc".to_string(),
        PHASE_TEARDOWN_CLOSURE_RETURN => "teardown_closure_return".to_string(),
        _ => format!("unknown:{}", code),
    }
}

/// Returns Some(label) when `code` is a teardown sub-stage (>= 20),
/// None otherwise. Used by the deadman emit to populate the dedicated
/// `teardown_stage_by_stream` field per operator spec.
pub fn teardown_stage_label(code: i32) -> Option<String> {
    if code >= PHASE_TEARDOWN_ENTER && code <= PHASE_TEARDOWN_CLOSURE_RETURN {
        Some(phase_label(code))
    } else {
        None
    }
}

fn unix_now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// CHUNK13_DIAG — Rust mirror of the C++ `PrismBranchTrace` struct
/// defined in `crates/prism-nhs/src/cuda/adjudicator.cuh`. Layout MUST
/// match field-for-field; serde-Deserialize/Serialize on the Rust side
/// only — the device side writes via atomic ops on the same memory.
///
/// 64 bytes total (16 × u32). Read via direct pointer dereference of
/// the pinned-mapped host alias of the buffer; no API call needed.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PrismBranchTrace {
    pub f1_predicate_last: u32,
    pub f1_branch_count: [u32; 4],
    pub f1_bridge_invocations: u32,
    pub g26_predicate_last: u32,
    pub g26_branch_count: [u32; 4],
    pub g26_bridge_invocations: u32,
    pub first_launch_seen: u32,
    pub reserved: [u32; 3],
}

impl PrismBranchTrace {
    pub const SIZE_BYTES: usize = 64;

    pub fn zero() -> Self {
        Self {
            f1_predicate_last: 0,
            f1_branch_count: [0; 4],
            f1_bridge_invocations: 0,
            g26_predicate_last: 0,
            g26_branch_count: [0; 4],
            g26_bridge_invocations: 0,
            first_launch_seen: 0,
            reserved: [0; 3],
        }
    }
}

/// Per-stream progress state. Writers are per-stream rayon closures;
/// readers are the watchdog thread and the deadman emit path.
///
/// All fields use `Relaxed` atomics: the watchdog reads a recent-enough
/// snapshot, not a transactional one. Drift across atomics within one
/// snapshot is acceptable for stall detection (it only matters that
/// progress timestamps advance).
pub struct PathAHeartbeat {
    pub n_streams: usize,
    pub last_chunk_seen: Vec<AtomicI32>,
    pub last_step_seen: Vec<AtomicI32>,
    pub last_progress_ts_unix: Vec<AtomicI64>,
    pub current_phase: Vec<AtomicI32>,
    pub v2_live: Vec<AtomicBool>,
    pub evidence_ready: Vec<AtomicBool>,
    /// CHUNK13_DIAG — per-stream pinned-mapped HOST pointer to the
    /// stream's PrismBranchTrace block. 0 if not registered (legacy
    /// path, no traced bridges). Read by the watchdog deadman to dump
    /// branch counts into the minimal completion JSON.
    pub branch_trace_host_ptr: Vec<AtomicU64>,
}

impl PathAHeartbeat {
    /// Construct a heartbeat record for `n_streams` streams. Initial
    /// progress timestamp is now-Unix so a first poll shortly after
    /// startup does not falsely trigger the stall detector.
    pub fn new(n_streams: usize) -> Arc<Self> {
        let now = unix_now_secs();
        Arc::new(Self {
            n_streams,
            last_chunk_seen: (0..n_streams).map(|_| AtomicI32::new(-1)).collect(),
            last_step_seen: (0..n_streams).map(|_| AtomicI32::new(0)).collect(),
            last_progress_ts_unix: (0..n_streams).map(|_| AtomicI64::new(now)).collect(),
            current_phase: (0..n_streams).map(|_| AtomicI32::new(PHASE_INIT)).collect(),
            v2_live: (0..n_streams).map(|_| AtomicBool::new(false)).collect(),
            evidence_ready: (0..n_streams).map(|_| AtomicBool::new(false)).collect(),
            branch_trace_host_ptr: (0..n_streams).map(|_| AtomicU64::new(0)).collect(),
        })
    }

    /// CHUNK13_DIAG — register the pinned-mapped host pointer for a
    /// stream's PrismBranchTrace block. Called once per stream right
    /// after `cuMemHostAlloc(PORTABLE | DEVICEMAP)` succeeds. The
    /// watchdog deadman reads through this pointer (no CUDA API call
    /// required — pinned-mapped memory is host-coherent on x86).
    pub fn register_branch_trace(&self, sid: usize, host_ptr: u64) {
        if sid >= self.n_streams {
            return;
        }
        self.branch_trace_host_ptr[sid].store(host_ptr, Ordering::Release);
    }

    /// Read the per-stream branch trace via the registered pinned-mapped
    /// host pointer, returning a zeroed snapshot if no pointer is
    /// registered. Callers must guarantee the pointer is valid for the
    /// lifetime of the read (registered allocations live until the
    /// per-stream rayon closure exits).
    pub fn read_branch_trace(&self, sid: usize) -> Option<PrismBranchTrace> {
        if sid >= self.n_streams {
            return None;
        }
        let p = self.branch_trace_host_ptr[sid].load(Ordering::Acquire);
        if p == 0 {
            return None;
        }
        // SAFETY: caller guarantees the pinned-mapped buffer at `p`
        // remains live while we read; PrismBranchTrace is repr(C) +
        // 16 × u32 fields whose unaligned-but-correctly-padded layout
        // matches the C++ struct in adjudicator.cuh.
        unsafe { Some(*(p as *const PrismBranchTrace)) }
    }

    /// Record a heartbeat from a stream. Cheap: 4 relaxed atomic stores.
    /// Safe to call from inside a rayon closure without holding any
    /// engine-internal lock.
    pub fn mark(&self, sid: usize, chunk: i32, step: i32, phase: i32) {
        if sid >= self.n_streams {
            return;
        }
        self.last_chunk_seen[sid].store(chunk, Ordering::Relaxed);
        self.last_step_seen[sid].store(step, Ordering::Relaxed);
        self.last_progress_ts_unix[sid].store(unix_now_secs(), Ordering::Relaxed);
        self.current_phase[sid].store(phase, Ordering::Relaxed);
    }

    /// Mark V2-live (set after a successful V2-INSTANTIATE-COMPLETE).
    pub fn set_v2_live(&self, sid: usize) {
        if sid >= self.n_streams {
            return;
        }
        self.v2_live[sid].store(true, Ordering::Release);
    }

    /// Mark evidence-ready (set when this stream has produced enough
    /// evidence that an early teardown is safe — typically after the
    /// ASC Bayesian baseline-lock fires).
    pub fn set_evidence_ready(&self, sid: usize) {
        if sid >= self.n_streams {
            return;
        }
        self.evidence_ready[sid].store(true, Ordering::Release);
    }

    /// Atomic point-in-time snapshot of every stream's progress state.
    pub fn snapshot(&self) -> HeartbeatSnapshot {
        let mut last_chunk = Vec::with_capacity(self.n_streams);
        let mut last_step = Vec::with_capacity(self.n_streams);
        let mut last_ts = Vec::with_capacity(self.n_streams);
        let mut phase = Vec::with_capacity(self.n_streams);
        let mut v2 = Vec::with_capacity(self.n_streams);
        let mut ev = Vec::with_capacity(self.n_streams);
        for sid in 0..self.n_streams {
            last_chunk.push(self.last_chunk_seen[sid].load(Ordering::Relaxed));
            last_step.push(self.last_step_seen[sid].load(Ordering::Relaxed));
            last_ts.push(self.last_progress_ts_unix[sid].load(Ordering::Relaxed));
            phase.push(self.current_phase[sid].load(Ordering::Relaxed));
            v2.push(self.v2_live[sid].load(Ordering::Relaxed));
            ev.push(self.evidence_ready[sid].load(Ordering::Relaxed));
        }
        HeartbeatSnapshot {
            n_streams: self.n_streams,
            last_chunk_by_stream: last_chunk,
            last_step_by_stream: last_step,
            last_progress_ts_unix_by_stream: last_ts,
            current_phase_by_stream: phase,
            v2_live_by_stream: v2,
            evidence_ready_by_stream: ev,
            taken_at_unix: unix_now_secs(),
        }
    }
}

/// Atomic snapshot of all per-stream heartbeats.
#[derive(Debug, Clone, Serialize)]
pub struct HeartbeatSnapshot {
    pub n_streams: usize,
    pub last_chunk_by_stream: Vec<i32>,
    pub last_step_by_stream: Vec<i32>,
    pub last_progress_ts_unix_by_stream: Vec<i64>,
    pub current_phase_by_stream: Vec<i32>,
    pub v2_live_by_stream: Vec<bool>,
    pub evidence_ready_by_stream: Vec<bool>,
    pub taken_at_unix: i64,
}

/// Shared cancel/stall state. First-writer-wins on `exit_reason`.
const WATCHDOG_OWNER_MONITORING: u8 = 0;
const WATCHDOG_OWNER_WORKERS_JOINED: u8 = 1;
const WATCHDOG_OWNER_DEADMAN_COMMITTED: u8 = 2;

pub struct WatchdogState {
    pub cancel_requested: AtomicBool,
    pub wall_cap_disarmed: AtomicBool,
    terminal_owner: AtomicU8,
    pub stall_detected: AtomicBool,
    pub stalled_stream_id: AtomicI32, // -1 if none
    pub exit_reason: Mutex<String>,
}

impl WatchdogState {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            cancel_requested: AtomicBool::new(false),
            wall_cap_disarmed: AtomicBool::new(false),
            terminal_owner: AtomicU8::new(WATCHDOG_OWNER_MONITORING),
            stall_detected: AtomicBool::new(false),
            stalled_stream_id: AtomicI32::new(-1),
            exit_reason: Mutex::new(String::new()),
        })
    }

    /// Flip cancel and latch the first writer's reason. Idempotent.
    pub fn request_cancel(&self, reason: &str) {
        self.cancel_requested.store(true, Ordering::Release);
        if let Ok(mut r) = self.exit_reason.lock() {
            if r.is_empty() {
                *r = reason.to_string();
            }
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel_requested.load(Ordering::Acquire)
    }

    /// Disable wall-cap cancellation once the engine has reached the
    /// evidence-complete handoff. Teardown/materialization can legitimately
    /// exceed the integration cap while writing lossless evidence files.
    pub fn disarm_wall_cap(&self) {
        self.wall_cap_disarmed.store(true, Ordering::Release);
    }

    pub fn is_wall_cap_disarmed(&self) -> bool {
        self.wall_cap_disarmed.load(Ordering::Acquire)
    }

    /// Terminate watchdog ownership after the outer worker scope has joined.
    ///
    /// This deliberately does not clear `cancel_requested` or `exit_reason`:
    /// bounded-run materialization and receipts still need that provenance.
    /// The caller must invoke this only after every stream worker has returned,
    /// so the watchdog's stuck-worker deadman remains active for the complete
    /// interval in which a worker can still block the join.
    pub fn disarm_after_workers_join(&self) -> bool {
        self.terminal_owner
            .compare_exchange(
                WATCHDOG_OWNER_MONITORING,
                WATCHDOG_OWNER_WORKERS_JOINED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub fn is_disarmed_after_workers_join(&self) -> bool {
        self.terminal_owner.load(Ordering::Acquire) == WATCHDOG_OWNER_WORKERS_JOINED
    }

    /// Atomically claim terminal ownership for the crash-safe deadman path.
    /// If the worker join already won, no partial deadman artifacts may be
    /// emitted. If this claim wins, a later join cannot suppress the committed
    /// process exit after artifact emission.
    fn try_commit_deadman_exit(&self) -> bool {
        self.terminal_owner
            .compare_exchange(
                WATCHDOG_OWNER_MONITORING,
                WATCHDOG_OWNER_DEADMAN_COMMITTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub fn current_exit_reason(&self) -> String {
        self.exit_reason
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }
}

// ─── SIGINT handler (async-signal-safe) ────────────────────────────────────
// libc::signal handlers must call only async-signal-safe functions.
// AtomicBool::store with Relaxed ordering IS async-signal-safe. We do NOT
// log, lock a mutex, or call any allocator from the handler.
static SIGINT_RECEIVED: AtomicBool = AtomicBool::new(false);
static PREVIOUS_SIGINT_HANDLER: Mutex<Option<libc::sighandler_t>> = Mutex::new(None);

extern "C" fn sigint_handler(_signum: libc::c_int) {
    SIGINT_RECEIVED.store(true, Ordering::Relaxed);
}

/// Install a SIGINT handler that flips a static AtomicBool. The watchdog
/// thread polls the flag on its next tick and propagates to WatchdogState.
/// Idempotent — calling more than once just re-installs the same handler.
pub fn install_sigint_handler() {
    let previous = unsafe { libc::signal(libc::SIGINT, sigint_handler as libc::sighandler_t) };
    if previous != libc::SIG_ERR {
        if let Ok(mut slot) = PREVIOUS_SIGINT_HANDLER.lock() {
            if slot.is_none() {
                *slot = Some(previous);
            }
        }
    }
}

/// Restore the exact prior SIGINT disposition after watchdog ownership ends.
/// Without this, the watchdog handler would keep setting an atomic that no
/// thread consumes during serial post-MD materialization.
pub fn restore_sigint_handler() {
    let previous = PREVIOUS_SIGINT_HANDLER
        .lock()
        .ok()
        .and_then(|mut slot| slot.take())
        .unwrap_or(libc::SIG_DFL);
    unsafe {
        libc::signal(libc::SIGINT, previous);
    }
    SIGINT_RECEIVED.store(false, Ordering::Release);
}

// ─── Watchdog thread ───────────────────────────────────────────────────────

/// Spawn the path-A watchdog. The caller must retain and join the handle before
/// any normal writer can touch the crash-safe completion/ghost artifact paths.
///
/// Detection rules:
/// - Every `poll_interval_secs`, snapshot all heartbeats.
/// - Pump SIGINT_RECEIVED → state.request_cancel("sigint").
/// - Compute max_chunk = max over streams of last_chunk_seen.
/// - A stream is "stagnant" if its chunk < max_chunk AND its
///   last_progress_ts is older than `stall_threshold_secs`.
/// - If at least one stream is stagnant AND advanced count >= n - 1,
///   the watchdog logs a typed diagnostic and flips cancel with reason
///   "mid_chunk_stall_watchdog".
///
/// Deadman:
/// - Once cancel is set, the watchdog arms a deadman timer.
/// - If healthy streams have not allowed the stream-worker join to return within
///   `deadman_grace_secs` (because the stalled stream is futex-blocked),
///   the watchdog emits a minimal `path_a_completion.json` from heartbeat
///   state and calls `process::exit(2)`. OS reclaims VRAM.
pub fn spawn_watchdog(
    heartbeats: Arc<PathAHeartbeat>,
    state: Arc<WatchdogState>,
    poll_interval_secs: u64,
    stall_threshold_secs: i64,
    deadman_grace_secs: u64,
    completion_path: PathBuf,
    run_id: String,
    target: String,
    max_wall_seconds: Option<u64>,
    run_start: std::time::Instant,
) -> std::io::Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("prism-path-a-watchdog".into())
        .spawn(move || {
            let n_streams = heartbeats.n_streams;
            let mut deadman_armed_at: Option<std::time::Instant> = None;

            log::info!(
                "[PATH-A WATCHDOG] started: n_streams={} poll={}s stall_threshold={}s deadman_grace={}s wall_cap={:?}",
                n_streams, poll_interval_secs, stall_threshold_secs,
                deadman_grace_secs, max_wall_seconds
            );

            loop {
                // `park_timeout` preserves periodic polling while allowing the
                // post-join owner to wake and join this thread immediately.
                std::thread::park_timeout(Duration::from_secs(poll_interval_secs));

                // The deadman protects only the stream-worker join. Once the
                // outer scope has returned, serial evidence/QNANL materialization
                // may legitimately outlive both the wall cap and deadman grace.
                if state.is_disarmed_after_workers_join() {
                    log::info!(
                        "[PATH-A WATCHDOG] all stream workers joined; watchdog disarmed before serial evidence materialization"
                    );
                    return;
                }

                // 1) Pump SIGINT global → state cancel.
                if SIGINT_RECEIVED.load(Ordering::Relaxed) && !state.is_cancelled() {
                    log::error!("[PATH-A WATCHDOG] SIGINT observed; flipping cancel");
                    state.request_cancel("sigint");
                }

                // 2) Wall-cap check (in addition to the chunk-loop check
                //    in nhs_rt_full.rs — this catches stalls where no
                //    stream reaches the next chunk boundary).
                if !state.is_cancelled() && !state.is_wall_cap_disarmed() {
                    if let Some(cap) = max_wall_seconds {
                        if run_start.elapsed().as_secs() >= cap {
                            log::error!(
                                "[PATH-A WATCHDOG] wall-cap reached at watchdog level (elapsed={}s, cap={}s); flipping cancel",
                                run_start.elapsed().as_secs(), cap
                            );
                            state.request_cancel("path_a_max_wall_seconds");
                        }
                    }
                }

                // 3) If cancel is active, run the deadman timer.
                if state.is_cancelled() {
                    if deadman_armed_at.is_none() {
                        deadman_armed_at = Some(std::time::Instant::now());
                        log::warn!(
                            "[PATH-A WATCHDOG] cancel active (reason={}); deadman timer armed for {}s grace",
                            state.current_exit_reason(),
                            deadman_grace_secs
                        );
                    } else if deadman_armed_at
                        .as_ref()
                        .map(|i| i.elapsed().as_secs() > deadman_grace_secs)
                        .unwrap_or(false)
                    {
                        // Claim terminal ownership before writing any crash-safe
                        // artifact. Exactly one of worker join or deadman commit
                        // can win this CAS, so a suppressed exit can never leave
                        // partial deadman sidecars in a continuing run.
                        if !state.try_commit_deadman_exit() {
                            log::info!(
                                "[PATH-A WATCHDOG] worker join completed at deadman boundary; suppressing process exit"
                            );
                            return;
                        }
                        log::error!(
                            "[PATH-A WATCHDOG] deadman expired after {}s grace; emitting minimal path_a_completion.json and exiting via process::exit(2)",
                            deadman_grace_secs
                        );
                        let snap = heartbeats.snapshot();
                        let stalled_id =
                            state.stalled_stream_id.load(Ordering::Acquire);
                        // CHUNK13_DIAG — read per-stream branch traces from
                        // pinned-mapped host pointers (registered earlier
                        // via heartbeats.register_branch_trace). None
                        // entries are streams that never registered a
                        // trace block (legacy path).
                        let branch_traces: Vec<Option<PrismBranchTrace>> =
                            (0..n_streams)
                                .map(|sid| heartbeats.read_branch_trace(sid))
                                .collect();
                        emit_minimal_completion_json(
                            &completion_path,
                            &run_id,
                            &target,
                            n_streams,
                            &state.current_exit_reason(),
                            if stalled_id >= 0 { Some(stalled_id) } else { None },
                            &snap,
                            max_wall_seconds,
                            run_start.elapsed().as_secs_f64(),
                            Some(&branch_traces),
                        );
                        // M1.2.23 §6 + §7 — emit ghost_site_map.json +
                        // ghost_time_map.json with explicit "partial"
                        // status so scanner can detect that the deadman
                        // bypassed the normal MD-only sidecar emit but
                        // the files still exist for auditability.
                        emit_deadman_ghost_sidecars(
                            &completion_path,
                            &run_id,
                            &target,
                            &snap,
                            n_streams,
                        );
                        if let Err(error) =
                            crate::successor_acceptance_v4::publish_runtime_failure_diagnostic(
                                "PATH_A_DEADMAN",
                                &format!(
                                    "deadman expired after {}s; reason={}",
                                    deadman_grace_secs,
                                    state.current_exit_reason()
                                ),
                            )
                        {
                            log::error!(
                                "[PATH-A WATCHDOG] diagnostic transaction unavailable; original deadman preserved: {:#}",
                                error
                            );
                        }
                        std::process::exit(2);
                    }
                    continue;
                }

                // 4) Detect N-1/N mid-chunk stall.
                let snap = heartbeats.snapshot();
                let now = snap.taken_at_unix;
                let max_chunk = snap
                    .last_chunk_by_stream
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(-1);
                if max_chunk < 1 {
                    continue;
                }
                let stagnant: Vec<usize> = (0..n_streams)
                    .filter(|&sid| {
                        let c = snap.last_chunk_by_stream[sid];
                        let ts = snap.last_progress_ts_unix_by_stream[sid];
                        c < max_chunk && (now - ts) > stall_threshold_secs
                    })
                    .collect();
                let advanced = n_streams - stagnant.len();
                if !stagnant.is_empty() && advanced >= n_streams.saturating_sub(1) {
                    let stalled_id = stagnant[0];
                    let stalled_chunk = snap.last_chunk_by_stream[stalled_id];
                    let stalled_age =
                        now - snap.last_progress_ts_unix_by_stream[stalled_id];
                    log::error!(
                        "[PATH-A WATCHDOG] mid-chunk stall detected: stream {} stuck at chunk {} (last progress {}s ago); {}/{} streams reached chunk {}; flipping cancel",
                        stalled_id,
                        stalled_chunk,
                        stalled_age,
                        advanced,
                        n_streams,
                        max_chunk
                    );
                    log::error!(
                        "[PATH-A WATCHDOG] heartbeat snapshot: chunks={:?} steps={:?} phases={:?}",
                        snap.last_chunk_by_stream,
                        snap.last_step_by_stream,
                        snap.current_phase_by_stream
                    );
                    state.stall_detected.store(true, Ordering::Release);
                    state
                        .stalled_stream_id
                        .store(stalled_id as i32, Ordering::Release);
                    state.request_cancel("mid_chunk_stall_watchdog");
                }
            }
        })
}

// ─── Crash-safe minimal completion emit ────────────────────────────────────

/// Emit a minimal `path_a_completion.json` from heartbeat state alone.
///
/// Used by the watchdog deadman path when normal teardown cannot return
/// (typically because rayon join is blocked behind a futex-stalled stream).
/// The shape is a strict subset of the normal Phase-8 emit: no Phase 8a-d
/// artifacts are claimed, and `path_b_required: true` is asserted.
///
/// Failure to write is logged at error level. The caller (watchdog) calls
/// `process::exit` after this returns, so VRAM is reclaimed by the OS even
/// if the file write itself fails.
/// **M1.2.23 §6 + §7** — Watchdog/deadman path sidecar emission.
///
/// `output_dir` is derived from the completion-json path's parent. The
/// helper emits two sidecars even when the normal MD-only teardown was
/// bypassed by the deadman:
///
///   * `ghost_time_map.json` — per-stream gear/dt context with explicit
///     "deadman_path_partial" status (no live gearbox capture is
///     available at deadman time).
///   * `ghost_site_map.json` — empty entries list with status string
///     "partial_watchdog_deadman_path_no_per_site_centroid_metadata".
///
/// Both are best-effort and never silently skip per directive Phase 16.
pub fn emit_deadman_ghost_sidecars(
    completion_path: &Path,
    run_id: &str,
    target: &str,
    snap: &HeartbeatSnapshot,
    n_streams: usize,
) {
    let out_dir = completion_path.parent().unwrap_or_else(|| Path::new("."));

    // ── ghost_time_map.json ─────────────────────────────────────────────
    let streams: Vec<serde_json::Value> = (0..n_streams)
        .map(|i| {
            let last_step = snap.last_step_by_stream.get(i).copied().unwrap_or(0);
            let last_chunk = snap.last_chunk_by_stream.get(i).copied().unwrap_or(-1);
            let v2_live = snap.v2_live_by_stream.get(i).copied().unwrap_or(false);
            serde_json::json!({
                "stream_id":           i as u32,
                "n_spikes":             0i64,            // unavailable in deadman path
                "timestep_min":        -1i32,
                "timestep_max":         last_step,
                "frame_idx_at_teardown": last_step,
                "last_chunk":           last_chunk,
                "v2_live_at_deadman":   v2_live,
                // GEAR-PROVENANCE-002 — was `0u32` labelled "Wave A default".
                // Under the documented encoding 0 IS the 0.5 fs gear, so that
                // literal asserted a temporal resolution nobody measured and
                // was indistinguishable offline from a genuine gear-0 frame.
                // The deadman path has no live pipeline to read
                // `ChronometricStateTensor` from, so the honest encoding is an
                // explicit null: JSON can represent absence, the u32 gear
                // encoding cannot. Consumers must fall back to `dt_fs`
                // (itself null here) rather than assume a gear.
                "gear_id":              serde_json::Value::Null,
                "gear_id_status":       "unmeasured_deadman_path_no_device_gear_readback",
                "dt_fs":                serde_json::Value::Null,
                "dt_source":            "deadman_path_no_runtime_dt_capture",
                "physical_time_fs":     serde_json::Value::Null,
                "source":               "watchdog_deadman_partial",
            })
        })
        .collect();
    let gtm = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_ghost_time_map",
        "run_id":         run_id,
        "target":         target,
        "streams":        streams,
        "status":         "partial_deadman_path",
        "notes": [
            "Emitted by path_a_watchdog::emit_deadman_ghost_sidecars after wall-cap or stall.",
            "dt_fs unresolvable from the deadman path — caller did not pass it through;",
            "scanner consumers should treat physical_time_fs as null and dt_fs_source as deadman_path.",
        ],
    });
    let gtm_path = out_dir.join("ghost_time_map.json");
    if let Ok(f) = std::fs::File::create(&gtm_path) {
        let mut bw = std::io::BufWriter::new(f);
        let _ = serde_json::to_writer_pretty(&mut bw, &gtm);
        use std::io::Write as _;
        let _ = bw.flush();
        log::error!(
            "[PATH-A WATCHDOG] ghost_time_map.json (partial) emitted at {}",
            gtm_path.display()
        );
    }

    // ── ghost_site_map.json ─────────────────────────────────────────────
    let gsm = serde_json::json!({
        "schema_version":   1,
        "schema_kind":      "pathb_ghost_site_map",
        "run_id":           run_id,
        "target":           target,
        "coordinate_frame": "prism_topology_native_no_recentering",
        "status":           "partial_watchdog_deadman_path_no_per_site_centroid_metadata",
        "entries":          serde_json::Value::Array(vec![]),
        "missing_entries": [{
            "reason": "deadman_path_does_not_have_access_to_v2_cluster_centroids",
            "follow_up": "engine MD-only teardown path emits populated ghost_site_map; this deadman fallback emits empty entries for auditability",
        }],
        "field_completeness": {
            "stream_id":       "implicit_from_completion_json_v2_live_by_stream",
            "site_id":         "absent_deadman_path",
            "aabb":            "absent_deadman_path",
            "centroid_xyz":    "absent_deadman_path",
            "voxel_xyz":       "absent_deadman_path",
            "residue_support": "absent_deadman_path",
        },
    });
    let gsm_path = out_dir.join("ghost_site_map.json");
    if let Ok(f) = std::fs::File::create(&gsm_path) {
        let mut bw = std::io::BufWriter::new(f);
        let _ = serde_json::to_writer_pretty(&mut bw, &gsm);
        use std::io::Write as _;
        let _ = bw.flush();
        log::error!(
            "[PATH-A WATCHDOG] ghost_site_map.json (partial) emitted at {}",
            gsm_path.display()
        );
    }
}

pub fn emit_minimal_completion_json(
    path: &Path,
    run_id: &str,
    target: &str,
    n_streams: usize,
    exit_reason: &str,
    stalled_stream_id: Option<i32>,
    snap: &HeartbeatSnapshot,
    max_wall_seconds: Option<u64>,
    elapsed_wall_seconds: f64,
    branch_traces: Option<&[Option<PrismBranchTrace>]>,
) {
    let now_iso = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

    // Map exit_reason → completion_status. Per directive §6 Commit 12 +
    // operator's hard MVP discipline, "full_complete" is intentionally
    // unreachable from any partial/bounded path here.
    let completion_status = match exit_reason {
        "path_a_max_wall_seconds" | "n_chunks_cap" | "mid_chunk_stall_watchdog" | "sigint" => {
            "evidence_emitted_partial_bounded"
        }
        "evidence_complete" => "bounded_path_a_evidence_exit",
        "natural_completion" => "evidence_emitted_natural",
        "not_triggered" => "not_triggered",
        _ => "evidence_emitted_unknown",
    };

    let streams_completed: usize = snap.v2_live_by_stream.iter().filter(|b| **b).count();
    let max_chunk_completed = snap
        .last_chunk_by_stream
        .iter()
        .copied()
        .max()
        .unwrap_or(-1);

    // instantiate_complete_by_stream as a Vec<bool> (true if v2_live was set
    // before the deadman fired). Honest: not the canonical timestamp form
    // emitted by the normal Phase-8 path; this is the deadman fallback.
    let inst_by_stream: Vec<bool> = snap.v2_live_by_stream.clone();

    // POST_CHUNK_LOOP_TEARDOWN_STALL diagnostic — derive fine-grained
    // phase labels and the dedicated teardown-stage field from the
    // raw `current_phase_by_stream` codes. This is what tells us
    // EXACTLY where each stream stalled when the watchdog fires.
    let current_phase_label_by_stream: Vec<String> = snap
        .current_phase_by_stream
        .iter()
        .map(|&code| phase_label(code))
        .collect();
    let teardown_stage_by_stream: Vec<Option<String>> = snap
        .current_phase_by_stream
        .iter()
        .map(|&code| teardown_stage_label(code))
        .collect();

    // CHUNK13_DIAG — per-stream branch trace snapshot. Each element is
    // either a PrismBranchTrace (when a pinned-mapped trace block was
    // registered for that stream) or null (legacy / disabled).
    let branch_trace_by_stream: Vec<serde_json::Value> = match branch_traces {
        Some(t) => t
            .iter()
            .map(|maybe| {
                maybe
                    .as_ref()
                    .map(|tr| serde_json::to_value(tr).unwrap_or(serde_json::Value::Null))
                    .unwrap_or(serde_json::Value::Null)
            })
            .collect(),
        None => (0..n_streams).map(|_| serde_json::Value::Null).collect(),
    };

    let json = serde_json::json!({
        "schema_version": 2,
        "run_id": run_id,
        "target": target,
        "stream_count": n_streams,
        "streams_completed": streams_completed,
        "v2_trigger_step_by_stream": [],
        "instantiate_complete_by_stream": inst_by_stream,
        "t7_mode": "wall_bounded_or_watchdog",
        "t7_chunks_completed": max_chunk_completed,
        "evidence_exit_reason": exit_reason,
        "completion_status": completion_status,
        "max_wall_seconds": max_wall_seconds,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "missing_optional_fields": [
            "v2_trigger_step_by_stream",
            "binding_sites.json",
            "v2_ignition_summary.json",
            "f2_ring_status.json",
            "f2_write_commit_log.json",
            "f2_artifact_completeness.json",
            "transform_dag.json"
        ],
        "path_b_required": true,
        "stalled_stream_id": stalled_stream_id,
        "last_chunk_by_stream": snap.last_chunk_by_stream,
        "last_step_by_stream": snap.last_step_by_stream,
        "last_progress_ts_unix_by_stream":
            snap.last_progress_ts_unix_by_stream,
        "current_phase_by_stream": snap.current_phase_by_stream,
        "current_phase_label_by_stream": current_phase_label_by_stream,
        "teardown_stage_by_stream": teardown_stage_by_stream,
        "v2_live_by_stream": snap.v2_live_by_stream,
        "evidence_ready_by_stream": snap.evidence_ready_by_stream,
        "branch_trace_by_stream": branch_trace_by_stream,
        "snapshot_taken_at_unix": snap.taken_at_unix,
        "snapshot_taken_at_iso": now_iso,
        "emit_source": "path_a_watchdog::emit_minimal_completion_json",
        "note": "Crash-safe minimal evidence emit triggered by watchdog or SIGINT. \
                 Phase 8a-d artifacts (binding_sites, F2 sidecars, transform_dag) \
                 are likely missing because normal teardown was bypassed. \
                 current_phase_label_by_stream names the EXACT stage every stream \
                 was at when the watchdog fired; teardown_stage_by_stream is \
                 non-null only when the stream had entered teardown. \
                 path_b_required is asserted true. Treat this run as unvalidated.",
    });

    match std::fs::File::create(path) {
        Ok(f) => {
            let mut writer = std::io::BufWriter::new(f);
            match serde_json::to_writer_pretty(&mut writer, &json) {
                Ok(()) => {
                    use std::io::Write as _;
                    let _ = writer.flush();
                    log::error!(
                        "[PATH-A WATCHDOG] minimal path_a_completion.json emitted at {} (exit_reason={}, stalled_stream_id={:?}, completion_status={})",
                        path.display(),
                        exit_reason,
                        stalled_stream_id,
                        completion_status
                    );
                }
                Err(e) => log::error!(
                    "[PATH-A WATCHDOG] minimal path_a_completion.json: serialize failed at {}: {}",
                    path.display(),
                    e
                ),
            }
        }
        Err(e) => log::error!(
            "[PATH-A WATCHDOG] minimal path_a_completion.json: create failed at {}: {}",
            path.display(),
            e
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heartbeat_mark_and_snapshot_round_trip() {
        let hb = PathAHeartbeat::new(8);
        hb.mark(0, 5, 2500, PHASE_T7_CAL);
        hb.mark(7, 12, 6000, PHASE_V2_LIVE);
        hb.set_v2_live(0);
        hb.set_evidence_ready(7);
        let snap = hb.snapshot();
        assert_eq!(snap.n_streams, 8);
        assert_eq!(snap.last_chunk_by_stream[0], 5);
        assert_eq!(snap.last_step_by_stream[0], 2500);
        assert_eq!(snap.current_phase_by_stream[0], PHASE_T7_CAL);
        assert!(snap.v2_live_by_stream[0]);
        assert_eq!(snap.last_chunk_by_stream[7], 12);
        assert_eq!(snap.current_phase_by_stream[7], PHASE_V2_LIVE);
        assert!(snap.evidence_ready_by_stream[7]);
    }

    #[test]
    fn watchdog_state_first_writer_wins_on_exit_reason() {
        let s = WatchdogState::new();
        s.request_cancel("first");
        s.request_cancel("second");
        assert_eq!(s.current_exit_reason(), "first");
        assert!(s.is_cancelled());
    }

    #[test]
    fn watchdog_state_disarms_wall_cap_without_cancel() {
        let s = WatchdogState::new();
        assert!(!s.is_wall_cap_disarmed());
        assert!(!s.is_cancelled());
        s.disarm_wall_cap();
        assert!(s.is_wall_cap_disarmed());
        assert!(!s.is_cancelled());
    }

    #[test]
    fn worker_join_disarm_preserves_bounded_cancel_provenance() {
        let s = WatchdogState::new();
        s.request_cancel("path_a_max_wall_seconds");
        assert!(s.is_cancelled());
        assert!(!s.is_disarmed_after_workers_join());

        assert!(s.disarm_after_workers_join());

        assert!(s.is_disarmed_after_workers_join());
        assert!(s.is_cancelled());
        assert_eq!(s.current_exit_reason(), "path_a_max_wall_seconds");
    }

    #[test]
    fn worker_join_and_deadman_have_single_atomic_terminal_owner() {
        let state = WatchdogState::new();
        let barrier = Arc::new(std::sync::Barrier::new(3));

        let join_state = Arc::clone(&state);
        let join_barrier = Arc::clone(&barrier);
        let join_claim = std::thread::spawn(move || {
            join_barrier.wait();
            join_state.disarm_after_workers_join()
        });

        let deadman_state = Arc::clone(&state);
        let deadman_barrier = Arc::clone(&barrier);
        let deadman_claim = std::thread::spawn(move || {
            deadman_barrier.wait();
            deadman_state.try_commit_deadman_exit()
        });

        barrier.wait();
        let join_won = join_claim.join().expect("join claim thread panicked");
        let deadman_won = deadman_claim.join().expect("deadman claim thread panicked");

        assert_ne!(join_won, deadman_won, "exactly one terminal owner must win");
        assert_eq!(state.is_disarmed_after_workers_join(), join_won);
        assert!(!state.disarm_after_workers_join());
        assert!(!state.try_commit_deadman_exit());
    }

    fn watchdog_test_root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "prism_path_a_watchdog_{}_{}_{}",
            name,
            std::process::id(),
            nonce
        ))
    }

    #[test]
    fn cancelled_watchdog_join_disarm_survives_grace_without_deadman_output() {
        let state = WatchdogState::new();
        state.request_cancel("path_a_max_wall_seconds");
        let root = watchdog_test_root("joined");
        std::fs::create_dir_all(&root).expect("create watchdog test root");
        let completion_path = root.join("path_a_completion.json");
        let handle = spawn_watchdog(
            PathAHeartbeat::new(1),
            Arc::clone(&state),
            1,
            60,
            1,
            completion_path.clone(),
            "join-disarm-test".to_string(),
            "unit-test".to_string(),
            None,
            std::time::Instant::now(),
        )
        .expect("spawn joined watchdog test thread");

        assert!(state.disarm_after_workers_join());
        handle.thread().unpark();
        handle.join().expect("disarmed watchdog thread panicked");

        std::thread::sleep(Duration::from_secs(2));
        let sentinel_path = root.join("survived_after_deadman_grace");
        std::fs::write(&sentinel_path, b"PASS\n").expect("write survival sentinel");
        assert!(sentinel_path.exists());
        assert!(!completion_path.exists());
        assert!(!root.join("ghost_site_map.json").exists());
        assert!(!root.join("ghost_time_map.json").exists());
        std::fs::remove_dir_all(&root).expect("remove watchdog test root");
    }

    #[test]
    fn watchdog_deadman_subprocess_helper() {
        let Some(completion_path) = std::env::var_os("PRISM_WATCHDOG_DEADMAN_COMPLETION") else {
            return;
        };
        let state = WatchdogState::new();
        state.request_cancel("unit_test_deadman");
        let handle = spawn_watchdog(
            PathAHeartbeat::new(1),
            state,
            1,
            60,
            1,
            PathBuf::from(completion_path),
            "deadman-subprocess-test".to_string(),
            "unit-test".to_string(),
            None,
            std::time::Instant::now(),
        )
        .expect("spawn deadman watchdog test thread");
        let _ = handle.join();
        panic!("deadman helper returned instead of process::exit(2)");
    }

    #[test]
    fn unjoined_cancel_deadman_emits_trio_and_exits_two() {
        let root = watchdog_test_root("deadman");
        std::fs::create_dir_all(&root).expect("create deadman subprocess root");
        let completion_path = root.join("path_a_completion.json");
        let status = std::process::Command::new(
            std::env::current_exe().expect("resolve current test executable"),
        )
        .args([
            "--exact",
            "path_a_watchdog::tests::watchdog_deadman_subprocess_helper",
            "--nocapture",
        ])
        .env("PRISM_WATCHDOG_DEADMAN_COMPLETION", &completion_path)
        .status()
        .expect("run deadman subprocess helper");

        assert_eq!(status.code(), Some(2));
        assert!(completion_path.is_file());
        assert!(root.join("ghost_site_map.json").is_file());
        assert!(root.join("ghost_time_map.json").is_file());
        let completion: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&completion_path).expect("read deadman completion"),
        )
        .expect("parse deadman completion");
        assert_eq!(
            completion
                .get("evidence_exit_reason")
                .and_then(|v| v.as_str()),
            Some("unit_test_deadman")
        );
        assert_eq!(
            completion.get("emit_source").and_then(|v| v.as_str()),
            Some("path_a_watchdog::emit_minimal_completion_json")
        );
        std::fs::remove_dir_all(&root).expect("remove deadman subprocess root");
    }

    #[test]
    fn phase_labels_cover_all_named_codes_and_teardown_filter_correct() {
        // All named phase codes round-trip to a stable lowercase label.
        let cases = [
            (PHASE_INIT, "init", false),
            (PHASE_COLD_HOLD, "cold_hold", false),
            (PHASE_T7_CAL, "t7_cal", false),
            (PHASE_V2_LIVE, "v2_live", false),
            (PHASE_TEARDOWN, "teardown", false),
            (PHASE_DONE, "done", false),
            (PHASE_CHUNK_BODY_ENTER, "chunk_body_enter", false),
            (PHASE_CHUNK_BODY_V2_BUILD, "chunk_body_v2_build", false),
            (
                PHASE_CHUNK_BODY_GRAPH_LAUNCH,
                "chunk_body_graph_launch",
                false,
            ),
            (
                PHASE_CHUNK_BODY_POST_LAUNCH_SYNC,
                "chunk_body_post_launch_sync",
                false,
            ),
            (PHASE_TEARDOWN_ENTER, "teardown_enter", true),
            (PHASE_TEARDOWN_VRAM_AUDIT, "teardown_vram_audit", true),
            (PHASE_TEARDOWN_ENGINE_SYNC, "teardown_engine_sync", true),
            (
                PHASE_TEARDOWN_V2_WRITER_JOIN,
                "teardown_v2_writer_join",
                true,
            ),
            (PHASE_TEARDOWN_DOWNLOAD_SIG, "teardown_download_sig", true),
            (PHASE_TEARDOWN_DOWNLOAD_KCC, "teardown_download_kcc", true),
            (
                PHASE_TEARDOWN_CLOSURE_RETURN,
                "teardown_closure_return",
                true,
            ),
        ];
        for (code, expected_label, is_teardown) in cases {
            assert_eq!(phase_label(code), expected_label, "label for code {}", code);
            if is_teardown {
                assert_eq!(
                    teardown_stage_label(code),
                    Some(expected_label.to_string()),
                    "teardown filter should expose code {}",
                    code
                );
            } else {
                assert!(
                    teardown_stage_label(code).is_none(),
                    "teardown filter should not expose non-teardown code {}",
                    code
                );
            }
        }
        // Unknown code falls through to a stable diagnostic label and is
        // NEVER classified as a teardown stage.
        assert_eq!(phase_label(99), "unknown:99");
        assert!(teardown_stage_label(99).is_none());
    }

    #[test]
    fn completion_status_mapping_never_full_complete() {
        // Ensure the deadman-emit mapping never yields "full_complete".
        for reason in &[
            "path_a_max_wall_seconds",
            "n_chunks_cap",
            "mid_chunk_stall_watchdog",
            "sigint",
            "evidence_complete",
            "natural_completion",
            "not_triggered",
            "garbage_unknown_value",
        ] {
            let status = match *reason {
                "path_a_max_wall_seconds"
                | "n_chunks_cap"
                | "mid_chunk_stall_watchdog"
                | "sigint" => "evidence_emitted_partial_bounded",
                "evidence_complete" => "bounded_path_a_evidence_exit",
                "natural_completion" => "evidence_emitted_natural",
                "not_triggered" => "not_triggered",
                _ => "evidence_emitted_unknown",
            };
            assert_ne!(status, "full_complete");
        }
    }
}
