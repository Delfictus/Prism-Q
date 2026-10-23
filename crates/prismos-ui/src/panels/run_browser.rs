//! Run-root browser panel.
//!
//! Lets the operator pick a PrismOS run root, lists its top-level entries
//! (directories first, then files, alphabetical), and runs `prismos doctor`
//! against the current root. The listing rebuilds reactively whenever the run
//! root changes, and a missing/unreadable root degrades to an inline
//! empty-state row rather than panicking.

use std::path::Path;
use std::rc::Rc;

use adw::prelude::*;

use crate::adapters::PrismCommand;
use crate::app::AppContext;
use crate::jobs::JobEvent;

/// Build the run-root browser panel. Returns the root widget.
pub fn build(ctx: &AppContext) -> gtk::Widget {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    root.add_css_class("prismos-root");

    // ---- Top bar: current run dir + choose + doctor -----------------------
    let top_bar = gtk::Box::new(gtk::Orientation::Horizontal, 12);
    top_bar.set_margin_top(12);
    top_bar.set_margin_bottom(12);
    top_bar.set_margin_start(12);
    top_bar.set_margin_end(12);

    let path_label = gtk::Label::new(Some(&current_dir_text(ctx)));
    path_label.add_css_class("mono");
    path_label.set_xalign(0.0);
    path_label.set_hexpand(true);
    path_label.set_ellipsize(gtk::pango::EllipsizeMode::Middle);
    path_label.set_selectable(true);

    let choose_button = gtk::Button::with_label("Choose run root...");
    choose_button.add_css_class("accent-button");

    let doctor_button = gtk::Button::with_label("Doctor");

    top_bar.append(&path_label);
    top_bar.append(&choose_button);
    top_bar.append(&doctor_button);
    root.append(&top_bar);

    root.append(&gtk::Separator::new(gtk::Orientation::Horizontal));

    // ---- Main area: scrolled list of entries ------------------------------
    let list = gtk::ListBox::new();
    list.set_selection_mode(gtk::SelectionMode::None);
    list.add_css_class("card");
    list.set_margin_top(12);
    list.set_margin_bottom(12);
    list.set_margin_start(12);
    list.set_margin_end(12);
    list.set_valign(gtk::Align::Start);

    let scroller = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vexpand(true)
        .hexpand(true)
        .child(&list)
        .build();
    root.append(&scroller);

    // Shared handles cloned into closures.
    let list = Rc::new(list);
    let path_label = Rc::new(path_label);

    // Initial population.
    populate_list(&list, ctx.run_dir().as_deref());

    // ---- Choose run root --------------------------------------------------
    {
        let ctx = ctx.clone();
        let list = list.clone();
        let path_label = path_label.clone();
        choose_button.connect_clicked(move |_btn| {
            let dialog = gtk::FileDialog::new();
            dialog.set_title("Choose run root");
            let window = ctx.window.clone();
            let ctx = ctx.clone();
            let list = list.clone();
            let path_label = path_label.clone();
            dialog.select_folder(Some(&window), gtk::gio::Cancellable::NONE, move |result| {
                if let Ok(folder) = result {
                    if let Some(path) = folder.path() {
                        ctx.set_run_dir(Some(path.clone()));
                        path_label.set_text(&path.display().to_string());
                        populate_list(&list, Some(path.as_path()));
                        ctx.toast(&format!("Run root: {}", path.display()));
                    }
                }
            });
        });
    }

    // ---- Doctor -----------------------------------------------------------
    {
        let ctx = ctx.clone();
        doctor_button.connect_clicked(move |_btn| {
            let run_dir = ctx.run_dir();
            let cmd = PrismCommand::os_doctor(run_dir.as_deref(), false);
            let ctx_done = ctx.clone();
            ctx.jobs.spawn(cmd, &ctx.resolver, move |ev| {
                if let JobEvent::Done(outcome) = ev {
                    if outcome.success {
                        ctx_done.toast("Doctor: PASS");
                    } else if let Some(err) = &outcome.error {
                        ctx_done.toast(&format!("Doctor failed to launch: {err}"));
                    } else {
                        let code = outcome
                            .code
                            .map(|c| c.to_string())
                            .unwrap_or_else(|| "signal".into());
                        ctx_done.toast(&format!("Doctor: FAILED (exit {code})"));
                    }
                }
            });
        });
    }

    root.upcast()
}

/// The display text for the current run root, or an empty-state message.
fn current_dir_text(ctx: &AppContext) -> String {
    match ctx.run_dir() {
        Some(path) => path.display().to_string(),
        None => "No run root selected".to_string(),
    }
}

/// Clear and (re)populate `list` from the entries of `dir`.
///
/// Directories sort first, then files, each alphabetically (case-insensitive).
/// A missing/unreadable/None directory yields a single inline empty-state row.
fn populate_list(list: &gtk::ListBox, dir: Option<&Path>) {
    // Remove all existing rows.
    while let Some(child) = list.first_child() {
        list.remove(&child);
    }

    let dir = match dir {
        Some(dir) => dir,
        None => {
            list.append(&empty_state_row("No run root selected"));
            return;
        }
    };

    let read = match std::fs::read_dir(dir) {
        Ok(read) => read,
        Err(err) => {
            list.append(&empty_state_row(&format!("Cannot read run root: {err}")));
            return;
        }
    };

    // Collect (is_dir, name, size_bytes) tuples.
    let mut entries: Vec<(bool, String, Option<u64>)> = Vec::new();
    for entry in read.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let (is_dir, size) = match entry.metadata() {
            Ok(meta) => {
                if meta.is_dir() {
                    (true, None)
                } else {
                    (false, Some(meta.len()))
                }
            }
            // Unreadable metadata: treat as a file with unknown size.
            Err(_) => (false, None),
        };
        entries.push((is_dir, name, size));
    }

    if entries.is_empty() {
        list.append(&empty_state_row("Run root is empty"));
        return;
    }

    // Directories first, then files; alphabetical (case-insensitive) within each.
    entries.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| a.1.to_lowercase().cmp(&b.1.to_lowercase()))
    });

    for (is_dir, name, size) in entries {
        list.append(&entry_row(is_dir, &name, size));
    }
}

/// Build one entry row: icon + name + dim right-aligned size.
fn entry_row(is_dir: bool, name: &str, size: Option<u64>) -> gtk::ListBoxRow {
    let row = gtk::ListBoxRow::new();
    let hbox = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    hbox.set_margin_top(6);
    hbox.set_margin_bottom(6);
    hbox.set_margin_start(10);
    hbox.set_margin_end(10);

    let icon_name = if is_dir {
        "folder-symbolic"
    } else {
        "text-x-generic-symbolic"
    };
    let icon = gtk::Image::from_icon_name(icon_name);
    hbox.append(&icon);

    let name_label = gtk::Label::new(Some(name));
    name_label.set_xalign(0.0);
    name_label.set_hexpand(true);
    name_label.set_ellipsize(gtk::pango::EllipsizeMode::Middle);
    if is_dir {
        name_label.add_css_class("mono");
    }
    hbox.append(&name_label);

    let size_text = if is_dir {
        "dir".to_string()
    } else {
        match size {
            Some(bytes) => format!("{bytes} B"),
            None => "—".to_string(),
        }
    };
    let size_label = gtk::Label::new(Some(&size_text));
    size_label.add_css_class("dim");
    size_label.set_xalign(1.0);
    hbox.append(&size_label);

    row.set_child(Some(&hbox));
    row
}

/// Build an inline empty-state / error row.
fn empty_state_row(message: &str) -> gtk::ListBoxRow {
    let row = gtk::ListBoxRow::new();
    row.set_selectable(false);
    row.set_activatable(false);

    let label = gtk::Label::new(Some(message));
    label.add_css_class("dim");
    label.set_xalign(0.0);
    label.set_margin_top(12);
    label.set_margin_bottom(12);
    label.set_margin_start(12);
    label.set_margin_end(12);

    row.set_child(Some(&label));
    row
}
