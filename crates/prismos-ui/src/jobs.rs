//! Async backend job runner.
//!
//! Spawns a typed [`PrismCommand`], streams its stdout/stderr **live** into the
//! global observability log, and delivers a terminal [`JobOutcome`] on the GTK
//! main thread. Uses a worker thread + `async-channel` drained by
//! `glib::spawn_future_local` — `glib::MainContext::channel` was removed in
//! glib 0.20, so this is the current idiom for thread→main-loop delivery.

use std::io::{BufRead, BufReader};
use std::os::unix::process::CommandExt;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

use gtk::prelude::*;

use crate::adapters::{BinaryResolver, PrismCommand};

/// A live event from a running backend job.
#[derive(Debug, Clone)]
pub enum JobEvent {
    Stdout(String),
    Stderr(String),
    Done(JobOutcome),
}

/// Terminal result of a backend job.
#[derive(Debug, Clone)]
pub struct JobOutcome {
    /// The exact resolved argv that ran (audit line).
    pub argv: String,
    /// Process exit code, or `None` if killed by signal / failed to spawn.
    pub code: Option<i32>,
    pub success: bool,
    /// Spawn error message, if the process could not be launched.
    pub error: Option<String>,
}

/// A cancel handle for a *cancellable* job (see [`JobRunner::spawn_cancellable`]).
///
/// The engine launched by `prism-validate-and-run.sh` is bash's grandchild
/// (the script `wait`s on `nhs_rt_full`; it does not `exec` it — see
/// `scripts/prism-validate-and-run.sh:374`). A plain `child.kill()` would reap
/// only bash and orphan the engine, which keeps burning the GPU. So the
/// cancellable spawn puts the whole tree in its own process group
/// (`process_group(0)`, so pgid == the child pid) and [`cancel`](Self::cancel)
/// signals the entire group.
#[derive(Clone)]
pub struct JobHandle {
    /// The child's process-group id, set once the process has spawned. `None`
    /// before spawn or after the job has completed.
    pgid: Arc<Mutex<Option<i32>>>,
    /// Leased controllers own native cleanup; signal their PID only, never their workers.
    controller_only: bool,
    controller_start_ticks: Arc<Mutex<Option<u64>>>,
    cancel_requested: Arc<AtomicBool>,
}

impl JobHandle {
    /// Signal the whole process group. `signal` is a `kill(1)` spec, e.g.
    /// `"-TERM"` for a graceful stop or `"-KILL"` for the escalation. No-op if
    /// the job has not spawned yet or has already finished.
    ///
    /// Uses `/bin/kill -- -<pgid>` (the leading `-` targets the group). There is
    /// no `libc`/`nix` dependency in this crate's Cargo.toml, so we shell out to
    /// the standard `kill` binary rather than call `killpg` directly.
    pub fn signal(&self, signal: &str) -> bool {
        if self.controller_only && signal == "-KILL" { return false; }
        if self.controller_only && signal == "-TERM" { self.cancel_requested.store(true, Ordering::SeqCst); }
        let pgid = *self.pgid.lock().unwrap();
        if self.controller_only {
            let expected = *self.controller_start_ticks.lock().unwrap();
            if expected.is_none() || pgid.and_then(controller_start_ticks) != expected { return false; }
        }
        match pgid {
            Some(pgid) if pgid > 0 => std::process::Command::new("/bin/kill")
                .arg(signal)
                .arg("--")
                .arg(if self.controller_only { pgid.to_string() } else { format!("-{pgid}") })
                .status()
                .map(|s| s.success())
                .unwrap_or(false),
            _ => false,
        }
    }

    /// Graceful cancel: send `SIGTERM` to the whole process group. The caller
    /// is expected to schedule a `-KILL` escalation (via `glib::timeout`) if the
    /// tree does not exit promptly.
    pub fn cancel(&self) -> bool {
        self.signal("-TERM")
    }

    /// Hard kill: send `SIGKILL` to the whole process group (the escalation).
    pub fn kill(&self) -> bool {
        self.signal("-KILL")
    }

    /// Whether the job is still running (process group is set).
    pub fn is_running(&self) -> bool {
        self.pgid.lock().unwrap().is_some()
    }
}

/// Read only the stable start identity; scheduling state is not incarnation identity.
fn controller_start_ticks(pid: i32) -> Option<u64> {
    let raw = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    raw.rsplit_once(')')?.1.split_whitespace().nth(19)?.parse().ok()
}

/// Cloneable handle to the job runner. All clones share one global log buffer,
/// which the Logs panel renders, so every backend invocation is observable in
/// one place regardless of which panel launched it.
#[derive(Clone)]
pub struct JobRunner {
    log: gtk::TextBuffer,
}

impl Default for JobRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl JobRunner {
    pub fn new() -> Self {
        Self {
            log: gtk::TextBuffer::new(None),
        }
    }

    /// The shared, append-only observability log buffer.
    pub fn log_buffer(&self) -> gtk::TextBuffer {
        self.log.clone()
    }

    fn append(&self, line: &str) {
        let mut end = self.log.end_iter();
        self.log.insert(&mut end, line);
        self.log.insert(&mut end, "\n");
    }

    /// Spawn `cmd`. Streams output to the global log and invokes `on_event` for
    /// each event on the GTK main thread; the final event is always `Done`.
    pub fn spawn<F>(&self, cmd: PrismCommand, resolver: &BinaryResolver, on_event: F)
    where
        F: Fn(JobEvent) + 'static,
    {
        let argv = cmd.display(resolver);
        let mut command = cmd.to_std_command(resolver);
        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        let (tx, rx) = async_channel::unbounded::<JobEvent>();
        let argv_thread = argv.clone();
        std::thread::spawn(move || match command.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                let tx_out = tx.clone();
                let out_handle = std::thread::spawn(move || {
                    if let Some(stdout) = stdout {
                        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                            let _ = tx_out.send_blocking(JobEvent::Stdout(line));
                        }
                    }
                });
                let tx_err = tx.clone();
                let err_handle = std::thread::spawn(move || {
                    if let Some(stderr) = stderr {
                        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                            let _ = tx_err.send_blocking(JobEvent::Stderr(line));
                        }
                    }
                });
                let _ = out_handle.join();
                let _ = err_handle.join();
                let code = child.wait().ok().and_then(|status| status.code());
                let _ = tx.send_blocking(JobEvent::Done(JobOutcome {
                    argv: argv_thread,
                    code,
                    success: code == Some(0),
                    error: None,
                }));
            }
            Err(err) => {
                let _ = tx.send_blocking(JobEvent::Done(JobOutcome {
                    argv: argv_thread,
                    code: None,
                    success: false,
                    error: Some(err.to_string()),
                }));
            }
        });

        self.append(&format!("$ {argv}"));
        let runner = self.clone();
        glib::spawn_future_local(async move {
            while let Ok(event) = rx.recv().await {
                match &event {
                    JobEvent::Stdout(line) => runner.append(line),
                    JobEvent::Stderr(line) => runner.append(&format!("[stderr] {line}")),
                    JobEvent::Done(outcome) => {
                        if let Some(err) = &outcome.error {
                            runner.append(&format!("[error] {err}"));
                        }
                        runner.append(&format!(
                            "[exit {}] {}",
                            outcome
                                .code
                                .map(|code| code.to_string())
                                .unwrap_or_else(|| "signal".into()),
                            if outcome.success { "OK" } else { "FAILED" }
                        ));
                    }
                }
                on_event(event);
            }
        });
    }

    /// Like [`spawn`](Self::spawn), but the whole process tree is placed in its
    /// own process group and a [`JobHandle`] is returned so the caller can
    /// cancel it (SIGTERM, then SIGKILL). Used by the Run Engine panel, whose
    /// backend (`prism-validate-and-run.sh`) `wait`s on the engine as a child —
    /// so cancelling must group-kill, not just kill bash.
    ///
    /// This is a deliberate copy of `spawn`'s body (rather than a refactor of
    /// it) so the existing, widely-used `spawn` path stays byte-for-byte
    /// unchanged. The only additions are `process_group(0)` before spawn and the
    /// pgid bookkeeping into the returned handle.
    pub fn spawn_cancellable<F>(
        &self,
        cmd: PrismCommand,
        resolver: &BinaryResolver,
        on_event: F,
    ) -> JobHandle
    where
        F: Fn(JobEvent) + 'static,
    {
        self.spawn_managed(cmd, resolver, on_event, false)
    }

    /// The production host owns durable stop, native/MPS cleanup, C1 sealing and lease release.
    /// UI cancellation signals only that controller and never escalates behind its back.
    pub fn spawn_controller<F>(&self, cmd: PrismCommand, resolver: &BinaryResolver, on_event: F) -> JobHandle
    where F: Fn(JobEvent) + 'static,
    {
        self.spawn_managed(cmd, resolver, on_event, true)
    }

    fn spawn_managed<F>(&self, cmd: PrismCommand, resolver: &BinaryResolver, on_event: F, controller_only: bool) -> JobHandle
    where F: Fn(JobEvent) + 'static,
    {
        let argv = cmd.display(resolver);
        let mut command = cmd.to_std_command(resolver);
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        // Put bash + nhs_rt_full in one new process group (pgid == child pid)
        // so cancel() can signal the whole tree, not just bash.
        command.process_group(0);

        let pgid: Arc<Mutex<Option<i32>>> = Arc::new(Mutex::new(None));
        let controller_start = Arc::new(Mutex::new(None));
        let cancel_requested = Arc::new(AtomicBool::new(false));
        let handle = JobHandle { pgid: pgid.clone(), controller_only,
            controller_start_ticks: controller_start.clone(), cancel_requested: cancel_requested.clone() };

        let (tx, rx) = async_channel::unbounded::<JobEvent>();
        let argv_thread = argv.clone();
        let pgid_thread = pgid.clone();
        std::thread::spawn(move || match command.spawn() {
            Ok(mut child) => {
                // Record the group id (== child pid, per process_group(0)).
                let child_pid = child.id() as i32;
                *controller_start.lock().unwrap() = controller_start_ticks(child_pid);
                *pgid_thread.lock().unwrap() = Some(child_pid);
                // A cancellation issued before spawn is retained. This exact
                // unreaped child is signalled only if its incarnation still matches.
                if controller_only && cancel_requested.load(Ordering::SeqCst) {
                    let expected = *controller_start.lock().unwrap();
                    if expected.is_some() && controller_start_ticks(child_pid) == expected {
                        let _ = std::process::Command::new("/bin/kill").args(["-TERM", "--", &child_pid.to_string()]).status();
                    }
                }
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                let tx_out = tx.clone();
                let out_handle = std::thread::spawn(move || {
                    if let Some(stdout) = stdout {
                        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                            let _ = tx_out.send_blocking(JobEvent::Stdout(line));
                        }
                    }
                });
                let tx_err = tx.clone();
                let err_handle = std::thread::spawn(move || {
                    if let Some(stderr) = stderr {
                        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                            let _ = tx_err.send_blocking(JobEvent::Stderr(line));
                        }
                    }
                });
                let _ = out_handle.join();
                let _ = err_handle.join();
                let code = child.wait().ok().and_then(|status| status.code());
                // Job finished: clear the pgid so cancel() becomes a no-op.
                *pgid_thread.lock().unwrap() = None;
                let _ = tx.send_blocking(JobEvent::Done(JobOutcome {
                    argv: argv_thread,
                    code,
                    success: code == Some(0),
                    error: None,
                }));
            }
            Err(err) => {
                *pgid_thread.lock().unwrap() = None;
                let _ = tx.send_blocking(JobEvent::Done(JobOutcome {
                    argv: argv_thread,
                    code: None,
                    success: false,
                    error: Some(err.to_string()),
                }));
            }
        });

        self.append(&format!("$ {argv}"));
        let runner = self.clone();
        glib::spawn_future_local(async move {
            while let Ok(event) = rx.recv().await {
                match &event {
                    JobEvent::Stdout(line) => runner.append(line),
                    JobEvent::Stderr(line) => runner.append(&format!("[stderr] {line}")),
                    JobEvent::Done(outcome) => {
                        if let Some(err) = &outcome.error {
                            runner.append(&format!("[error] {err}"));
                        }
                        runner.append(&format!(
                            "[exit {}] {}",
                            outcome
                                .code
                                .map(|code| code.to_string())
                                .unwrap_or_else(|| "signal".into()),
                            if outcome.success { "OK" } else { "FAILED" }
                        ));
                    }
                }
                on_event(event);
            }
        });

        handle
    }
}
