//! Native diagnostics panel.
//!
//! Drives the read-only health-probe backends — hardware inspection, the CUDA
//! self-test, installed-OS control status, runtime attestation, and the
//! run-root doctor — each as a typed [`PrismCommand`] spawned through the shared
//! [`crate::jobs::JobRunner`]. Output streams live into this panel's own
//! monospace transcript (cleared per invocation) and, because every spawn also
//! tees into the global observability buffer, simultaneously into the Logs
//! panel. No process is ever launched directly and no shell is involved.

use std::path::Path;

use adw::prelude::*;

use crate::adapters::PrismCommand;
use crate::app::AppContext;
use crate::jobs::JobEvent;

/// Build the native diagnostics panel.
pub fn build(ctx: &AppContext) -> gtk::Widget {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 12);
    root.add_css_class("prismos-root");
    root.set_margin_top(16);
    root.set_margin_bottom(16);
    root.set_margin_start(16);
    root.set_margin_end(16);

    // ---- Header ------------------------------------------------------------
    let title = gtk::Label::new(Some("Native Diagnostics"));
    title.add_css_class("prismos-title");
    title.set_halign(gtk::Align::Start);
    root.append(&title);

    let subtitle = gtk::Label::new(Some(
        "Read-only health probes. Output streams here and into the global Logs panel.",
    ));
    subtitle.add_css_class("subtitle");
    subtitle.set_halign(gtk::Align::Start);
    subtitle.set_wrap(true);
    root.append(&subtitle);

    // ---- Transcript view (built first so buttons can clone its buffer) -----
    let transcript = gtk::TextView::new();
    transcript.set_editable(false);
    transcript.set_cursor_visible(false);
    transcript.set_monospace(true);
    transcript.set_wrap_mode(gtk::WrapMode::None);
    transcript.add_css_class("mono");
    transcript.add_css_class("log-view");
    let buffer = transcript.buffer();
    buffer.set_text("Select a diagnostic to run.\n");

    // ---- Action row (horizontal button row) --------------------------------
    let actions = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    actions.set_halign(gtk::Align::Start);

    add_action(&actions, ctx, &buffer, "Hardware inspect", |_ctx| {
        PrismCommand::hw_inspect(true, None)
    });
    add_action(&actions, ctx, &buffer, "CUDA selftest", |_ctx| {
        PrismCommand::hw_cuda_selftest(None)
    });
    add_action(&actions, ctx, &buffer, "Control status", |ctx| {
        match ctx.run_dir() {
            Some(dir) => PrismCommand::ctl_status(&dir),
            None => PrismCommand::ctl_status(Path::new(".")),
        }
    });
    add_action(&actions, ctx, &buffer, "Runtime attest", |_ctx| {
        PrismCommand::attest_runtime(None)
    });
    add_action(&actions, ctx, &buffer, "OS doctor", |ctx| {
        PrismCommand::os_doctor(ctx.run_dir().as_deref(), false)
    });

    root.append(&actions);

    // ---- Scrolled transcript (expands to fill) -----------------------------
    let scroller = gtk::ScrolledWindow::builder()
        .hexpand(true)
        .vexpand(true)
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&transcript)
        .build();
    scroller.add_css_class("card");
    root.append(&scroller);

    root.upcast()
}

/// Create one action button and append it to the flow row.
///
/// `make_cmd` builds the typed command fresh on each click (so it observes the
/// current run root). Clicking clears this panel's transcript, writes the
/// resolved argv as a `$ `-prefixed header, then streams the job's output into
/// the same buffer and toasts the terminal outcome.
fn add_action<F>(
    actions: &gtk::Box,
    ctx: &AppContext,
    buffer: &gtk::TextBuffer,
    label: &str,
    make_cmd: F,
) where
    F: Fn(&AppContext) -> PrismCommand + 'static,
{
    let button = gtk::Button::with_label(label);
    button.add_css_class("accent-button");

    let ctx = ctx.clone();
    let buffer = buffer.clone();
    let actions_guard = actions.clone();
    button.connect_clicked(move |_| {
        // In-flight guard: disable the whole action row until this job reaches
        // its terminal Done event, so a second click cannot wipe the transcript
        // mid-stream and interleave two jobs' output into one buffer.
        actions_guard.set_sensitive(false);

        // Build the command once: derive the display line, then hand the same
        // command to the runner.
        let cmd = make_cmd(&ctx);
        let argv = cmd.display(&ctx.resolver);

        // Fresh transcript for this invocation.
        buffer.set_text("");
        append_line(&buffer, &format!("$ {argv}"));

        let buf_events = buffer.clone();
        let ctx_events = ctx.clone();
        let actions_done = actions_guard.clone();
        ctx.jobs.spawn(cmd, &ctx.resolver, move |ev| match ev {
            JobEvent::Stdout(line) => append_line(&buf_events, &line),
            JobEvent::Stderr(line) => append_line(&buf_events, &format!("[stderr] {line}")),
            JobEvent::Done(outcome) => {
                if let Some(err) = &outcome.error {
                    append_line(&buf_events, &format!("[error] {err}"));
                }
                let code = outcome
                    .code
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "signal".to_string());
                append_line(
                    &buf_events,
                    &format!(
                        "[exit {}] {}",
                        code,
                        if outcome.success { "OK" } else { "FAILED" }
                    ),
                );
                if outcome.success {
                    ctx_events.toast("Diagnostic completed");
                } else {
                    ctx_events.toast("Diagnostic failed");
                }
                actions_done.set_sensitive(true);
            }
        });
    });

    actions.append(&button);
}

/// Append a line of text (plus a trailing newline) to a transcript buffer.
fn append_line(buffer: &gtk::TextBuffer, text: &str) {
    let mut end = buffer.end_iter();
    buffer.insert(&mut end, text);
    buffer.insert(&mut end, "\n");
}
