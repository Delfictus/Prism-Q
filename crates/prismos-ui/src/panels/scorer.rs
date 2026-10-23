//! Native Scorer panel — GFlowNet-coupling molecule scorer + lead-optimization report.
//!
//! Paste one or more SMILES (one per line). "Batch Score" ranks them all by the
//! Oracle-aligned reward (Mamba-2 student + reward head) with percentile/rank vs
//! the real 130k scored corpus. "Full Report" takes the first molecule and builds
//! the complete lead-optimization report: grounded RDKit chemistry + corpus
//! analogs + champion Oracle physics + gpt-oss Companion design hypotheses, with
//! explicit markers for any field that needs an MD/docking/FEP run to populate.

use adw::prelude::*;

use crate::adapters::PrismCommand;
use crate::app::AppContext;
use crate::jobs::JobEvent;

/// Build the native Scorer panel.
pub fn build(ctx: &AppContext) -> gtk::Widget {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 12);
    root.add_css_class("prismos-root");
    root.set_margin_top(16);
    root.set_margin_bottom(16);
    root.set_margin_start(16);
    root.set_margin_end(16);

    let title = gtk::Label::new(Some("PRISM Scorer — Binding Reward & Lead Optimization"));
    title.add_css_class("prismos-title");
    title.set_halign(gtk::Align::Start);
    root.append(&title);

    let subtitle = gtk::Label::new(Some(
        "Paste one SMILES per line (optionally \"id,smiles\"). Batch Score ranks them all by the \
         Oracle-aligned reward with percentile/rank vs the 130k scored corpus. Full Report builds \
         the complete lead-optimization report for the first molecule.",
    ));
    subtitle.add_css_class("subtitle");
    subtitle.set_halign(gtk::Align::Start);
    subtitle.set_wrap(true);
    root.append(&subtitle);

    // Multi-line input for one-or-many molecules.
    let input = gtk::TextView::new();
    input.set_monospace(true);
    input.set_wrap_mode(gtk::WrapMode::WordChar);
    input.add_css_class("mono");
    input.set_left_margin(6);
    input.set_top_margin(6);
    let input_buf = input.buffer();
    input_buf.set_text("Cc1ccc2[nH]cc(CC[C@@H]3CNCC3=O)c2c1");
    let input_scroll = gtk::ScrolledWindow::builder()
        .height_request(90)
        .hexpand(true)
        .child(&input)
        .build();
    input_scroll.add_css_class("card");
    root.append(&input_scroll);

    let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let batch_btn = gtk::Button::with_label("Batch Score");
    batch_btn.add_css_class("accent-button");
    let report_btn = gtk::Button::with_label("Full Report (first molecule)");
    row.append(&batch_btn);
    row.append(&report_btn);
    root.append(&row);

    let transcript = gtk::TextView::new();
    transcript.set_editable(false);
    transcript.set_cursor_visible(false);
    transcript.set_monospace(true);
    transcript.set_wrap_mode(gtk::WrapMode::None);
    transcript.add_css_class("mono");
    transcript.add_css_class("log-view");
    let out_buf = transcript.buffer();
    out_buf.set_text("Enter molecules above, then Batch Score or Full Report.\n");

    let scroller = gtk::ScrolledWindow::builder()
        .hexpand(true)
        .vexpand(true)
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&transcript)
        .build();
    scroller.add_css_class("card");
    root.append(&scroller);

    {
        let ctx = ctx.clone();
        let input_buf = input_buf.clone();
        let out_buf = out_buf.clone();
        let batch_c = batch_btn.clone();
        let report_c = report_btn.clone();
        batch_btn.connect_clicked(move |_| {
            let lines = collect_lines(&input_buf);
            if lines.is_empty() {
                return;
            }
            // SMILES never contain commas, so a comma-joined list is a safe payload.
            let csv = lines.join(",");
            run(
                &ctx,
                &out_buf,
                PrismCommand::score_batch(&csv),
                &[&batch_c, &report_c],
            );
        });
    }
    {
        let ctx = ctx.clone();
        let input_buf = input_buf.clone();
        let out_buf = out_buf.clone();
        let batch_c = batch_btn.clone();
        let report_c = report_btn.clone();
        report_btn.connect_clicked(move |_| {
            let lines = collect_lines(&input_buf);
            let Some(first) = lines.first() else { return };
            // For "id,smiles" lines, the report takes the SMILES field.
            let smiles = first
                .split(',')
                .next_back()
                .unwrap_or(first)
                .trim()
                .to_string();
            run(
                &ctx,
                &out_buf,
                PrismCommand::score_report(&smiles),
                &[&batch_c, &report_c],
            );
        });
    }

    root.upcast()
}

/// Non-empty, trimmed lines from the input buffer.
fn collect_lines(buf: &gtk::TextBuffer) -> Vec<String> {
    let text = buf
        .text(&buf.start_iter(), &buf.end_iter(), false)
        .to_string();
    text.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

fn run(ctx: &AppContext, out_buf: &gtk::TextBuffer, cmd: PrismCommand, buttons: &[&gtk::Button]) {
    for b in buttons {
        b.set_sensitive(false);
    }
    let argv = cmd.display(&ctx.resolver);
    out_buf.set_text("");
    append_line(out_buf, &format!("$ {argv}"));
    append_line(
        out_buf,
        "(scoring — first call warms the corpus index / Companion)\n",
    );

    let buf_events = out_buf.clone();
    let ctx_events = ctx.clone();
    let buttons_owned: Vec<gtk::Button> = buttons.iter().map(|b| (*b).clone()).collect();
    ctx.jobs.spawn(cmd, &ctx.resolver, move |ev| match ev {
        JobEvent::Stdout(line) => append_line(&buf_events, &line),
        JobEvent::Stderr(line) => append_line(&buf_events, &format!("[stderr] {line}")),
        JobEvent::Done(outcome) => {
            if let Some(err) = &outcome.error {
                append_line(&buf_events, &format!("[error] {err}"));
            }
            append_line(
                &buf_events,
                &format!("[{}]", if outcome.success { "OK" } else { "FAILED" }),
            );
            ctx_events.toast(if outcome.success {
                "Scoring complete"
            } else {
                "Scoring failed"
            });
            for b in &buttons_owned {
                b.set_sensitive(true);
            }
        }
    });
}

fn append_line(buffer: &gtk::TextBuffer, text: &str) {
    let mut end = buffer.end_iter();
    buffer.insert(&mut end, text);
    buffer.insert(&mut end, "\n");
}
