//! QNANL receipt encode / decode / header panel.
//!
//! Three `card` sections drive the `prismqnanl` backend through the typed
//! [`crate::adapters::PrismCommand`] constructors — there is no shell and no
//! direct `std::process` use here: every invocation goes through
//! `ctx.jobs.spawn`, which owns the worker threads and streams stdout/stderr
//! live into the shared global Logs buffer. Each card surfaces its terminal
//! success/failure as a toast; the full transcript is always in the Logs panel.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use adw::prelude::*;

use crate::adapters::PrismCommand;
use crate::app::AppContext;
use crate::jobs;

/// Build the QNANL panel root (a scrolled column of three cards).
pub fn build(ctx: &AppContext) -> gtk::Widget {
    let content = gtk::Box::new(gtk::Orientation::Vertical, 18);
    content.set_margin_top(18);
    content.set_margin_bottom(18);
    content.set_margin_start(18);
    content.set_margin_end(18);

    content.append(&build_encode_card(ctx));
    content.append(&build_decode_card(ctx));
    content.append(&build_header_card(ctx));

    let scroller = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .child(&content)
        .build();
    scroller.set_hexpand(true);
    scroller.set_vexpand(true);
    scroller.upcast()
}

// ------------------------------------------------------------------------
// ENCODE card
// ------------------------------------------------------------------------

fn build_encode_card(ctx: &AppContext) -> gtk::Box {
    let card = new_card(
        "Encode",
        "Seal an input artifact into a QNANL packet + receipt.",
    );

    // Input artifact picker — path captured in an Rc<RefCell<Option<PathBuf>>>.
    let input_store = add_file_picker_row(
        &card,
        &ctx.window,
        "Choose input…",
        "No input artifact selected",
    );

    let family_entry = add_labeled_entry(&card, "Tokenizer family", "SpikeEvent");
    let packet_entry = add_labeled_entry(&card, "Output packet path", "");
    packet_entry.set_placeholder_text(Some("e.g. out.prismqnanl"));
    let receipt_entry = add_labeled_entry(&card, "Output receipt path", "");
    receipt_entry.set_placeholder_text(Some("e.g. out.receipt.json"));

    let encode_btn = gtk::Button::with_label("Encode");
    encode_btn.add_css_class("accent-button");
    encode_btn.set_halign(gtk::Align::Start);
    card.append(&encode_btn);

    encode_btn.connect_clicked({
        let ctx = ctx.clone();
        let input_store = input_store.clone();
        let family_entry = family_entry.clone();
        let packet_entry = packet_entry.clone();
        let receipt_entry = receipt_entry.clone();
        move |_| {
            let input = input_store.borrow().clone();
            let Some(input) = input else {
                ctx.toast("Select an input artifact before encoding");
                return;
            };

            let packet_text = packet_entry.text();
            let receipt_text = receipt_entry.text();
            if packet_text.trim().is_empty() || receipt_text.trim().is_empty() {
                ctx.toast("Provide both an output packet and receipt path");
                return;
            }

            let family_text = family_entry.text();
            let family = family_text.trim();
            let family = if family.is_empty() {
                "SpikeEvent"
            } else {
                family
            };

            let cmd = PrismCommand::qnanl_encode(
                &input,
                Path::new(packet_text.as_str()),
                Path::new(receipt_text.as_str()),
                family,
                None,
            );
            run_command(
                &ctx,
                cmd,
                "QNANL encode complete",
                "QNANL encode failed — see Logs",
            );
        }
    });

    card
}

// ------------------------------------------------------------------------
// DECODE card
// ------------------------------------------------------------------------

fn build_decode_card(ctx: &AppContext) -> gtk::Box {
    let card = new_card(
        "Decode",
        "Materialize a QNANL packet back into its source artifact.",
    );

    let packet_store =
        add_file_picker_row(&card, &ctx.window, "Choose packet…", "No packet selected");

    let output_entry = add_labeled_entry(&card, "Output path", "");
    output_entry.set_placeholder_text(Some("e.g. decoded.json"));

    let decode_btn = gtk::Button::with_label("Decode");
    decode_btn.set_halign(gtk::Align::Start);
    card.append(&decode_btn);

    decode_btn.connect_clicked({
        let ctx = ctx.clone();
        let packet_store = packet_store.clone();
        let output_entry = output_entry.clone();
        move |_| {
            let packet = packet_store.borrow().clone();
            let Some(packet) = packet else {
                ctx.toast("Select a QNANL packet before decoding");
                return;
            };

            let output_text = output_entry.text();
            if output_text.trim().is_empty() {
                ctx.toast("Provide an output path for the decoded artifact");
                return;
            }

            let cmd = PrismCommand::qnanl_decode(&packet, Path::new(output_text.as_str()));
            run_command(
                &ctx,
                cmd,
                "QNANL decode complete",
                "QNANL decode failed — see Logs",
            );
        }
    });

    card
}

// ------------------------------------------------------------------------
// HEADER card
// ------------------------------------------------------------------------

fn build_header_card(ctx: &AppContext) -> gtk::Box {
    let card = new_card(
        "Header",
        "Inspect a QNANL packet header without decoding the payload.",
    );

    let packet_store =
        add_file_picker_row(&card, &ctx.window, "Choose packet…", "No packet selected");

    let header_btn = gtk::Button::with_label("Show header");
    header_btn.set_halign(gtk::Align::Start);
    card.append(&header_btn);

    header_btn.connect_clicked({
        let ctx = ctx.clone();
        let packet_store = packet_store.clone();
        move |_| {
            let packet = packet_store.borrow().clone();
            let Some(packet) = packet else {
                ctx.toast("Select a QNANL packet to inspect");
                return;
            };

            let cmd = PrismCommand::qnanl_header(&packet);
            run_command(
                &ctx,
                cmd,
                "QNANL header read complete",
                "QNANL header read failed — see Logs",
            );
        }
    });

    card
}

// ------------------------------------------------------------------------
// Shared builders
// ------------------------------------------------------------------------

/// A titled `card` container with a dim subtitle.
fn new_card(title: &str, subtitle: &str) -> gtk::Box {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 10);
    card.add_css_class("card");

    let title_label = gtk::Label::new(Some(title));
    title_label.set_halign(gtk::Align::Start);
    title_label.add_css_class("prismos-title");
    card.append(&title_label);

    let subtitle_label = gtk::Label::new(Some(subtitle));
    subtitle_label.set_halign(gtk::Align::Start);
    subtitle_label.set_xalign(0.0);
    subtitle_label.set_wrap(true);
    subtitle_label.add_css_class("subtitle");
    subtitle_label.add_css_class("dim");
    card.append(&subtitle_label);

    card
}

/// Append a `label + Entry` row to `card`, returning the entry.
fn add_labeled_entry(card: &gtk::Box, label: &str, default: &str) -> gtk::Entry {
    let caption = gtk::Label::new(Some(label));
    caption.set_halign(gtk::Align::Start);
    caption.add_css_class("subtitle");
    card.append(&caption);

    let entry = gtk::Entry::new();
    entry.set_hexpand(true);
    if !default.is_empty() {
        entry.set_text(default);
    }
    card.append(&entry);
    entry
}

/// Append a file-picker row (button + mono path label) to `card`. The chosen
/// path is stored in the returned `Rc<RefCell<Option<PathBuf>>>` so command
/// closures can read it later. Uses the async `gtk::FileDialog` (GTK 4.10).
fn add_file_picker_row(
    card: &gtk::Box,
    window: &adw::ApplicationWindow,
    button_label: &str,
    empty_text: &str,
) -> Rc<RefCell<Option<PathBuf>>> {
    let store: Rc<RefCell<Option<PathBuf>>> = Rc::new(RefCell::new(None));

    let row = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    let button = gtk::Button::with_label(button_label);
    button.set_valign(gtk::Align::Center);

    let path_label = gtk::Label::new(Some(empty_text));
    path_label.set_halign(gtk::Align::Start);
    path_label.set_xalign(0.0);
    path_label.set_wrap(true);
    path_label.set_hexpand(true);
    path_label.add_css_class("mono");
    path_label.add_css_class("dim");

    row.append(&button);
    row.append(&path_label);
    card.append(&row);

    button.connect_clicked({
        let window = window.clone();
        let store = store.clone();
        let path_label = path_label.clone();
        move |_| {
            let dialog = gtk::FileDialog::new();
            let store = store.clone();
            let path_label = path_label.clone();
            dialog.open(Some(&window), gtk::gio::Cancellable::NONE, move |result| {
                if let Ok(file) = result {
                    if let Some(path) = file.path() {
                        path_label.set_text(&path.display().to_string());
                        path_label.remove_css_class("dim");
                        *store.borrow_mut() = Some(path);
                    }
                }
            });
        }
    });

    store
}

/// Spawn `cmd` through the shared job runner and toast its terminal outcome.
/// All stdout/stderr already streams into the global Logs buffer.
fn run_command(ctx: &AppContext, cmd: PrismCommand, ok_msg: &'static str, fail_msg: &'static str) {
    let ctx_done = ctx.clone();
    ctx.jobs.spawn(cmd, &ctx.resolver, move |ev| {
        if let jobs::JobEvent::Done(outcome) = ev {
            ctx_done.toast(if outcome.success { ok_msg } else { fail_msg });
        }
    });
}
