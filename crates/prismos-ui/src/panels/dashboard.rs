//! Operational dashboard panel.
//!
//! Two stacked cards inside a scroller:
//!
//! 1. **Overall status** — parses the durable `prismcode_status.json` gate object
//!    (run-root first, then the workspace fallback) with the tolerant
//!    [`crate::model::status::StatusDoc`] parser and renders a big native status
//!    pill, the schema version, and a per-lane PASS/BLOCKED breakdown. When no
//!    status source exists yet it degrades to a dim empty-state line instead of
//!    failing.
//! 2. **Operator / Science view** — drives the installed `prismui` dashboard
//!    binary through [`crate::jobs::JobRunner`] (worker thread, no shell) and
//!    streams its stdout live into a monospace text view, toasting the terminal
//!    success/failure outcome.
//!
//! Every backend invocation goes through `ctx.jobs.spawn`; nothing here touches
//! `std::process` directly and nothing blocks the GTK main loop.

use std::path::PathBuf;

use adw::prelude::*;

use crate::adapters::PrismCommand;
use crate::app::AppContext;
use crate::jobs::JobEvent;
use crate::model::status::StatusDoc;

/// Build the operational dashboard panel root widget.
pub fn build(ctx: &AppContext) -> gtk::Widget {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 12);
    root.set_margin_top(16);
    root.set_margin_bottom(16);
    root.set_margin_start(16);
    root.set_margin_end(16);

    let title = gtk::Label::new(Some("Operational Dashboard"));
    title.set_halign(gtk::Align::Start);
    title.add_css_class("prismos-title");
    root.append(&title);

    let subtitle = gtk::Label::new(Some(
        "PrismOS Beta2.0 — native runtime readiness gate and operator / science surface.",
    ));
    subtitle.set_halign(gtk::Align::Start);
    subtitle.set_wrap(true);
    subtitle.add_css_class("subtitle");
    root.append(&subtitle);

    // ---------------------------------------------------------------- (1) ---
    // Overall status card. The content box is rebuilt in-place by the refresh
    // button so a freshly written status object is picked up without a restart.
    let status_card = gtk::Box::new(gtk::Orientation::Vertical, 10);
    status_card.add_css_class("card");

    let status_header = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let status_heading = gtk::Label::new(Some("Overall status"));
    status_heading.set_halign(gtk::Align::Start);
    status_heading.set_hexpand(true);
    status_heading.add_css_class("subtitle");
    status_header.append(&status_heading);

    let refresh_btn = gtk::Button::with_label("Refresh status");
    refresh_btn.set_halign(gtk::Align::End);
    status_header.append(&refresh_btn);
    status_card.append(&status_header);

    let status_content = gtk::Box::new(gtk::Orientation::Vertical, 8);
    status_card.append(&status_content);
    root.append(&status_card);

    // Initial render.
    populate_status(&status_content, ctx);

    {
        let ctx_refresh = ctx.clone();
        let content = status_content.clone();
        refresh_btn.connect_clicked(move |_| {
            populate_status(&content, &ctx_refresh);
            ctx_refresh.toast("Status refreshed");
        });
    }

    // ---------------------------------------------------------------- (2) ---
    // Operator / Science view card.
    let view_card = gtk::Box::new(gtk::Orientation::Vertical, 10);
    view_card.add_css_class("card");

    let view_header = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let view_heading = gtk::Label::new(Some("Operator / Science view"));
    view_heading.set_halign(gtk::Align::Start);
    view_heading.set_hexpand(true);
    view_heading.add_css_class("subtitle");
    view_header.append(&view_heading);

    let run_btn = gtk::Button::with_label("Run prismui dashboard");
    run_btn.add_css_class("accent-button");
    run_btn.set_halign(gtk::Align::End);
    view_header.append(&run_btn);
    view_card.append(&view_header);

    let text_view = gtk::TextView::new();
    text_view.set_editable(false);
    text_view.set_cursor_visible(false);
    text_view.set_monospace(true);
    text_view.set_wrap_mode(gtk::WrapMode::WordChar);
    text_view.add_css_class("mono");
    text_view.add_css_class("log-view");

    let text_scroll = gtk::ScrolledWindow::builder()
        .min_content_height(280)
        .vexpand(true)
        .child(&text_view)
        .build();
    view_card.append(&text_scroll);
    root.append(&view_card);

    // Idle hint until the operator runs the binary.
    text_view.buffer().set_text(
        "Press \u{201c}Run prismui dashboard\u{201d} to render the operator / science view.\n",
    );

    {
        let ctx_run = ctx.clone();
        let buffer = text_view.buffer();
        run_btn.connect_clicked(move |_| {
            buffer.set_text("");

            // Pass the same status source the native gate uses, when present.
            let status_path = find_status_path(&ctx_run);

            let buffer_evt = buffer.clone();
            let ctx_done = ctx_run.clone();
            ctx_run.jobs.spawn(
                PrismCommand::ui_dashboard(None, status_path.as_deref(), None, None),
                &ctx_run.resolver,
                move |event| match event {
                    JobEvent::Stdout(line) => {
                        append_line(&buffer_evt, &line);
                    }
                    JobEvent::Stderr(line) => {
                        append_line(&buffer_evt, &format!("[stderr] {line}"));
                    }
                    JobEvent::Done(outcome) => {
                        if outcome.success {
                            ctx_done.toast("prismui dashboard completed (OK)");
                        } else {
                            let detail = outcome.error.clone().unwrap_or_else(|| {
                                outcome
                                    .code
                                    .map(|code| format!("exit {code}"))
                                    .unwrap_or_else(|| "terminated by signal".to_string())
                            });
                            append_line(&buffer_evt, &format!("[failed] {detail}"));
                            ctx_done.toast(&format!("prismui dashboard failed: {detail}"));
                        }
                    }
                },
            );
        });
    }

    let scroller = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vexpand(true)
        .hexpand(true)
        .child(&root)
        .build();

    scroller.upcast()
}

/// Append `text` followed by a newline to the end of `buffer`.
fn append_line(buffer: &gtk::TextBuffer, text: &str) {
    let mut end = buffer.end_iter();
    buffer.insert(&mut end, text);
    buffer.insert(&mut end, "\n");
}

/// First existing status source: explicit configuration, the active run root,
/// then the installed global status root.
fn find_status_path(ctx: &AppContext) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(configured) = std::env::var("PRISMOS_UI_STATUS_PATH") {
        candidates.push(PathBuf::from(configured));
    }
    if let Some(run_dir) = ctx.run_dir() {
        candidates.push(run_dir.join("build/status/prismcode_status.json"));
        candidates.push(run_dir.join("status/prismcode_status.json"));
    }
    candidates.push(PathBuf::from(
        "/var/lib/prismos/status/prismcode_status.json",
    ));
    candidates.into_iter().find(|path| path.is_file())
}

/// Clear and re-render the overall-status card content.
fn populate_status(container: &gtk::Box, ctx: &AppContext) {
    clear_children(container);

    let Some(path) = find_status_path(ctx) else {
        container.append(&dim_label("No status source found yet"));
        container.append(&dim_label(
            "Expected prismcode_status.json under the active run or /var/lib/prismos/status.",
        ));
        return;
    };

    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) => {
            container.append(&dim_label(&format!(
                "Could not read {}: {err}",
                path.display()
            )));
            return;
        }
    };

    let doc = match StatusDoc::from_json_str(&text) {
        Ok(doc) => doc,
        Err(err) => {
            container.append(&dim_label(&format!(
                "Could not parse {}: {err}",
                path.display()
            )));
            return;
        }
    };

    // Big overall pill + source.
    let overall = doc.overall();
    let pass = overall == "PASS";
    let header_row = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    header_row.append(&big_pill(overall, pass));
    let source = dim_label(&format!("source: {}", path.display()));
    source.set_valign(gtk::Align::Center);
    source.set_wrap(false);
    source.add_css_class("mono");
    header_row.append(&source);
    container.append(&header_row);

    // Schema version subtitle.
    let schema = gtk::Label::new(Some(doc.schema()));
    schema.set_halign(gtk::Align::Start);
    schema.add_css_class("dim");
    schema.add_css_class("mono");
    container.append(&schema);

    // Validation lanes subsection.
    let lanes_title = gtk::Label::new(Some("Validation lanes"));
    lanes_title.set_halign(gtk::Align::Start);
    lanes_title.set_margin_top(6);
    lanes_title.add_css_class("subtitle");
    container.append(&lanes_title);

    if doc.lanes.is_empty() {
        container.append(&dim_label("No lanes reported."));
    } else {
        for lane in &doc.lanes {
            let row = gtk::Box::new(gtk::Orientation::Horizontal, 10);

            let lane_pass = lane.is_pass();
            row.append(&pill(lane.status_text(), lane_pass));

            let key = gtk::Label::new(Some(&lane.key));
            key.set_halign(gtk::Align::Start);
            key.add_css_class("mono");
            row.append(&key);

            if let Some(evidence) = &lane.evidence {
                let evidence_label = gtk::Label::new(Some(&truncate(evidence, 90)));
                evidence_label.set_halign(gtk::Align::Start);
                evidence_label.set_hexpand(true);
                evidence_label.set_xalign(0.0);
                evidence_label.add_css_class("dim");
                row.append(&evidence_label);
            }

            container.append(&row);
        }
    }

    // Blocking lanes summary (only when something is actually blocking).
    let blocking = doc.blocking_lanes();
    if !blocking.is_empty() {
        let blocking_title = gtk::Label::new(Some("Blocking lanes"));
        blocking_title.set_halign(gtk::Align::Start);
        blocking_title.set_margin_top(6);
        blocking_title.add_css_class("subtitle");
        container.append(&blocking_title);

        for entry in &blocking {
            let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
            row.append(&pill("BLOCKED", false));
            let label = gtk::Label::new(Some(entry));
            label.set_halign(gtk::Align::Start);
            label.add_css_class("mono");
            row.append(&label);
            container.append(&row);
        }
    }
}

/// Remove every child of `container`.
fn clear_children(container: &gtk::Box) {
    while let Some(child) = container.first_child() {
        container.remove(&child);
    }
}

/// A compact status pill: `pass` (green) when `pass`, otherwise `blocked` (red).
fn pill(text: &str, pass: bool) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.set_halign(gtk::Align::Start);
    label.set_valign(gtk::Align::Center);
    label.add_css_class("status-pill");
    label.add_css_class(if pass { "pass" } else { "blocked" });
    label
}

/// The large overall verdict pill, rendered with emphasis via Pango markup.
fn big_pill(text: &str, pass: bool) -> gtk::Label {
    let label = gtk::Label::new(None);
    label.set_markup(&format!(
        "<span size='x-large' weight='bold'>{}</span>",
        glib::markup_escape_text(text)
    ));
    label.set_halign(gtk::Align::Start);
    label.set_valign(gtk::Align::Center);
    label.add_css_class("status-pill");
    label.add_css_class(if pass { "pass" } else { "blocked" });
    label
}

/// A wrapping, dimmed informational label.
fn dim_label(text: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.set_halign(gtk::Align::Start);
    label.set_xalign(0.0);
    label.set_wrap(true);
    label.add_css_class("dim");
    label
}

/// Truncate `text` to at most `max` characters, appending an ellipsis when cut.
fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{head}\u{2026}")
}
