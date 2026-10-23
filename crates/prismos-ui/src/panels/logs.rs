//! Logs / command-execution observability panel.
//!
//! Renders the single shared, append-only observability log owned by
//! [`crate::jobs::JobRunner`]. Every backend invocation across every panel
//! streams its resolved argv, live stdout/stderr, and terminal exit line into
//! one [`gtk::TextBuffer`]; this panel is the canonical view of that buffer.
//!
//! The view binds the *existing* shared buffer (never a fresh one) so it stays
//! in sync no matter which panel launched the job. The buffer's `changed`
//! signal drives autoscroll-to-bottom via a right-gravity end mark.

use adw::prelude::*;

use crate::app::AppContext;

/// Build the logs panel root widget.
pub fn build(ctx: &AppContext) -> gtk::Widget {
    // Bind the shared global observability buffer — do NOT create a new one.
    let buffer = ctx.jobs.log_buffer();

    let view = gtk::TextView::with_buffer(&buffer);
    view.set_editable(false);
    view.set_cursor_visible(false);
    view.set_monospace(true);
    view.set_wrap_mode(gtk::WrapMode::WordChar);
    view.set_left_margin(8);
    view.set_right_margin(8);
    view.set_top_margin(6);
    view.set_bottom_margin(6);
    view.add_css_class("mono");
    view.add_css_class("log-view");

    // Toolbar row: dim title + Clear + Copy all.
    let toolbar = gtk::Box::new(gtk::Orientation::Horizontal, 8);

    let title = gtk::Label::new(Some("Command execution log"));
    title.add_css_class("dim");
    title.set_halign(gtk::Align::Start);
    title.set_hexpand(true);
    toolbar.append(&title);

    let clear_button = gtk::Button::with_label("Clear");
    let copy_button = gtk::Button::with_label("Copy all");
    toolbar.append(&clear_button);
    toolbar.append(&copy_button);

    // Clear empties the shared buffer in place.
    let buffer_for_clear = buffer.clone();
    clear_button.connect_clicked(move |_button| {
        buffer_for_clear.set_text("");
    });

    // Copy all reads the full buffer text and pushes it to the clipboard.
    let buffer_for_copy = buffer.clone();
    let ctx_for_copy = ctx.clone();
    copy_button.connect_clicked(move |_button| {
        let text = buffer_for_copy.text(
            &buffer_for_copy.start_iter(),
            &buffer_for_copy.end_iter(),
            false,
        );
        ctx_for_copy.window.clipboard().set_text(text.as_str());
        ctx_for_copy.toast("Log copied to clipboard");
    });

    // Scrollable log surface that fills the panel.
    let scroller = gtk::ScrolledWindow::new();
    scroller.set_hexpand(true);
    scroller.set_vexpand(true);
    scroller.set_hscrollbar_policy(gtk::PolicyType::Automatic);
    scroller.set_vscrollbar_policy(gtk::PolicyType::Automatic);
    scroller.set_child(Some(&view));
    scroller.add_css_class("card");

    // Autoscroll: a right-gravity mark pinned at the end tracks fresh output;
    // on every buffer change we scroll it onscreen. Deferred to idle so the
    // newly inserted line is laid out before we scroll to it.
    let end_mark = buffer.create_mark(None, &buffer.end_iter(), false);
    let view_for_scroll = view.clone();
    buffer.connect_changed(move |_buffer| {
        let view = view_for_scroll.clone();
        let mark = end_mark.clone();
        glib::idle_add_local_once(move || {
            view.scroll_mark_onscreen(&mark);
        });
    });

    let root = gtk::Box::new(gtk::Orientation::Vertical, 8);
    root.set_margin_top(12);
    root.set_margin_bottom(12);
    root.set_margin_start(12);
    root.set_margin_end(12);
    root.append(&toolbar);
    root.append(&scroller);

    root.upcast()
}
