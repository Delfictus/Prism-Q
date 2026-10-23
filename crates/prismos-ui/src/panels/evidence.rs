//! Evidence / artifact explorer panel.
//!
//! A two-pane browser over the artifacts that live under the currently selected
//! run root. The **left** pane walks `ctx.run_dir()` up to two directory levels
//! deep (a bounded, symlink-safe `read_dir` recursion) and lists every
//! `.json` / `.jsonl` / `.prismqnanl` / `.tsv` file it finds. The **right** pane
//! shows the selected artifact's path, byte size, a monospace text preview
//! (first ~400 lines for text formats; a note for binary QNANL packets), and two
//! backend actions:
//!
//! * **QNANL header** — `prismqnanl header --packet <path>` (enabled only for
//!   `.prismqnanl` packets), and
//! * **Verify seal** — `prismseal verify <path>`.
//!
//! Both run through [`crate::jobs::JobRunner::spawn`], so their output streams
//! into the shared global Logs panel and a terminal toast reports success or
//! failure. Nothing here blocks the main loop, calls a shell, or panics on a
//! read/parse error.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use adw::prelude::*;

use crate::adapters::PrismCommand;
use crate::app::AppContext;
use crate::jobs::JobEvent;

/// Maximum directory recursion below the run root (root contents = depth 0).
const MAX_DEPTH: usize = 2;
/// Hard cap on the number of artifacts listed, to stay responsive on huge trees.
const MAX_FILES: usize = 5000;
/// Maximum number of text lines rendered in the inline preview.
const MAX_PREVIEW_LINES: usize = 400;
/// Files larger than this are not slurped into the preview view.
const MAX_PREVIEW_BYTES: u64 = 8 * 1024 * 1024;

/// Explanatory text shown in the preview pane for binary QNANL packets.
const QNANL_NOTE: &str = "\
This is a binary QNANL packet (.prismqnanl).

It is content-addressed and zstd-sealed, so it has no human-readable text
body to show inline. Use the 'QNANL header' button above to decode and
inspect its header / receipt metadata via the prismqnanl backend — the
output is streamed live into the global Logs panel.";

/// Build the evidence panel. The body is wrapped in a container that rebuilds
/// on every `map` (each time the panel becomes visible), so a run-root change
/// made in the Run Root panel is reflected the next time Evidence is shown.
pub fn build(ctx: &AppContext) -> gtk::Widget {
    let container = gtk::Box::new(gtk::Orientation::Vertical, 0);
    container.set_hexpand(true);
    container.set_vexpand(true);
    let ctx_map = ctx.clone();
    container.connect_map(move |c| populate(c, &ctx_map));
    container.upcast()
}

/// Clear `container` and rebuild the evidence body from the current run root.
fn populate(container: &gtk::Box, ctx: &AppContext) {
    while let Some(child) = container.first_child() {
        container.remove(&child);
    }
    container.append(&build_body(ctx));
}

fn build_body(ctx: &AppContext) -> gtk::Widget {
    // Resolve the run root. No run root (or a stale, non-directory one) is an
    // empty state, not an error.
    let run_dir = match ctx.run_dir() {
        Some(dir) if dir.is_dir() => dir,
        _ => {
            return empty_state(
                "No run root selected",
                "Choose a run root in the Run Root panel to browse its evidence \
                 artifacts (.json, .jsonl, .prismqnanl, .tsv).",
            )
        }
    };

    // Walk the tree once, up front, on the main thread (cheap directory I/O).
    let mut files: Vec<PathBuf> = Vec::new();
    collect_artifacts(&run_dir, 0, &mut files);

    if files.is_empty() {
        return empty_state(
            "No artifacts found",
            "No .json, .jsonl, .prismqnanl, or .tsv files were found within two \
             directory levels of the selected run root.",
        );
    }

    // --- Shared panel-local state -------------------------------------------
    // Row index -> absolute path. Read-only after construction, but held behind
    // Rc<RefCell<…>> per the panel contract so the selection handler can index
    // it by `ListBoxRow::index()`.
    let file_index: Rc<RefCell<Vec<PathBuf>>> = Rc::new(RefCell::new(files.clone()));
    // The currently selected artifact, shared with the action-button handlers.
    let selected: Rc<RefCell<Option<PathBuf>>> = Rc::new(RefCell::new(None));

    // --- LEFT: artifact list -------------------------------------------------
    let list = gtk::ListBox::new();
    list.set_selection_mode(gtk::SelectionMode::Single);
    list.add_css_class("navigation-sidebar");

    for path in &files {
        list.append(&list_row(path, &run_dir));
    }

    let list_scroll = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&list)
        .vexpand(true)
        .width_request(320)
        .build();
    list_scroll.add_css_class("sidebar-pane");

    // --- RIGHT: artifact details --------------------------------------------
    let path_label = gtk::Label::new(Some("Select an artifact"));
    path_label.set_halign(gtk::Align::Start);
    path_label.set_xalign(0.0);
    path_label.set_wrap(true);
    path_label.set_selectable(true);
    path_label.add_css_class("mono");

    let size_label = gtk::Label::new(Some(""));
    size_label.set_halign(gtk::Align::Start);
    size_label.set_xalign(0.0);
    size_label.add_css_class("dim");

    let header_btn = gtk::Button::with_label("QNANL header");
    header_btn.set_sensitive(false);
    header_btn.add_css_class("accent-button");

    let verify_btn = gtk::Button::with_label("Verify seal");
    verify_btn.set_sensitive(false);

    let actions = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    actions.append(&header_btn);
    actions.append(&verify_btn);

    let preview_caption = gtk::Label::new(Some("Preview"));
    preview_caption.set_halign(gtk::Align::Start);
    preview_caption.add_css_class("subtitle");

    let preview_view = gtk::TextView::new();
    preview_view.set_editable(false);
    preview_view.set_cursor_visible(false);
    preview_view.set_monospace(true);
    preview_view.set_wrap_mode(gtk::WrapMode::None);
    preview_view.add_css_class("mono");
    preview_view.add_css_class("log-view");
    let preview_buffer = preview_view.buffer();
    preview_buffer.set_text("Select an artifact from the list to preview its contents.");

    let preview_scroll = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&preview_view)
        .hexpand(true)
        .vexpand(true)
        .build();

    let details = gtk::Box::new(gtk::Orientation::Vertical, 10);
    details.set_margin_top(14);
    details.set_margin_bottom(14);
    details.set_margin_start(14);
    details.set_margin_end(14);
    details.append(&path_label);
    details.append(&size_label);
    details.append(&actions);
    details.append(&preview_caption);
    details.append(&preview_scroll);
    details.add_css_class("card");

    // --- Selection wiring ----------------------------------------------------
    {
        let file_index = file_index.clone();
        let selected = selected.clone();
        let path_label = path_label.clone();
        let size_label = size_label.clone();
        let header_btn = header_btn.clone();
        let verify_btn = verify_btn.clone();
        let preview_buffer = preview_buffer.clone();
        list.connect_row_selected(move |_list, row| {
            let Some(row) = row else {
                return;
            };
            let idx = row.index();
            if idx < 0 {
                return;
            }
            let path = {
                let files = file_index.borrow();
                match files.get(idx as usize) {
                    Some(path) => path.clone(),
                    None => return,
                }
            };

            *selected.borrow_mut() = Some(path.clone());

            path_label.set_text(&path.to_string_lossy());
            let size_text = match std::fs::metadata(&path) {
                Ok(meta) => human_bytes(meta.len()),
                Err(_) => "size unavailable".to_string(),
            };
            size_label.set_text(&size_text);

            let is_qnanl = ext_lower(&path) == "prismqnanl";
            header_btn.set_sensitive(is_qnanl);
            verify_btn.set_sensitive(true);

            preview_buffer.set_text(&load_preview(&path));
        });
    }

    // --- 'QNANL header' action ----------------------------------------------
    {
        let ctx = ctx.clone();
        let selected = selected.clone();
        header_btn.connect_clicked(move |_| {
            let path = match selected.borrow().clone() {
                Some(path) => path,
                None => return,
            };
            let ctx_done = ctx.clone();
            ctx.jobs.spawn(
                PrismCommand::qnanl_header(&path),
                &ctx.resolver,
                move |event| {
                    if let JobEvent::Done(outcome) = event {
                        ctx_done.toast(if outcome.success {
                            "QNANL header: OK — see Logs"
                        } else {
                            "QNANL header: FAILED — see Logs"
                        });
                    }
                },
            );
        });
    }

    // --- 'Verify seal' action ------------------------------------------------
    {
        let ctx = ctx.clone();
        let selected = selected.clone();
        verify_btn.connect_clicked(move |_| {
            let path = match selected.borrow().clone() {
                Some(path) => path,
                None => return,
            };
            let ctx_done = ctx.clone();
            ctx.jobs.spawn(
                PrismCommand::seal_verify(&path),
                &ctx.resolver,
                move |event| {
                    if let JobEvent::Done(outcome) = event {
                        ctx_done.toast(if outcome.success {
                            "Verify seal: OK — see Logs"
                        } else {
                            "Verify seal: FAILED — see Logs"
                        });
                    }
                },
            );
        });
    }

    // --- Assemble the split layout ------------------------------------------
    let paned = gtk::Paned::new(gtk::Orientation::Horizontal);
    paned.set_wide_handle(true);
    paned.set_start_child(Some(&list_scroll));
    paned.set_end_child(Some(&details));
    paned.set_resize_start_child(false);
    paned.set_shrink_start_child(false);
    paned.set_shrink_end_child(false);
    paned.set_position(360);
    paned.set_hexpand(true);
    paned.set_vexpand(true);

    // Auto-select the first artifact so the right pane is never blank.
    if let Some(first) = list.row_at_index(0) {
        list.select_row(Some(&first));
    }

    paned.upcast()
}

/// One left-pane row: bold basename + dim path relative to the run root.
fn list_row(path: &Path, run_dir: &Path) -> gtk::ListBoxRow {
    let basename = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned());
    let relative = path
        .strip_prefix(run_dir)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned();

    let name_label = gtk::Label::new(Some(&basename));
    name_label.set_halign(gtk::Align::Start);
    name_label.set_xalign(0.0);
    name_label.set_ellipsize(gtk::pango::EllipsizeMode::Middle);
    name_label.add_css_class("mono");

    let rel_label = gtk::Label::new(Some(&relative));
    rel_label.set_halign(gtk::Align::Start);
    rel_label.set_xalign(0.0);
    rel_label.set_ellipsize(gtk::pango::EllipsizeMode::Start);
    rel_label.add_css_class("dim");

    let row_box = gtk::Box::new(gtk::Orientation::Vertical, 2);
    row_box.set_margin_top(4);
    row_box.set_margin_bottom(4);
    row_box.set_margin_start(8);
    row_box.set_margin_end(8);
    row_box.append(&name_label);
    row_box.append(&rel_label);

    let row = gtk::ListBoxRow::new();
    row.set_child(Some(&row_box));
    row
}

/// Centered empty-state page (no run root, or no matching artifacts).
fn empty_state(title: &str, description: &str) -> gtk::Widget {
    adw::StatusPage::builder()
        .title(title)
        .description(description)
        .icon_name("dialog-information-symbolic")
        .build()
        .upcast()
}

/// Recursively collect artifact files under `dir`, bounded by [`MAX_DEPTH`] and
/// [`MAX_FILES`]. Uses the dirent file type so symlinks are never followed,
/// which also makes the walk cycle-safe. Never panics: unreadable directories
/// and entries are skipped.
fn collect_artifacts(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if out.len() >= MAX_FILES {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    let mut sub_dirs: Vec<PathBuf> = Vec::new();
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(_) => continue,
        };
        let path = entry.path();
        if file_type.is_dir() {
            sub_dirs.push(path);
        } else if file_type.is_file() && is_artifact(&path) {
            files.push(path);
        }
    }

    files.sort();
    for file in files {
        if out.len() >= MAX_FILES {
            return;
        }
        out.push(file);
    }

    if depth < MAX_DEPTH {
        sub_dirs.sort();
        for sub_dir in sub_dirs {
            collect_artifacts(&sub_dir, depth + 1, out);
            if out.len() >= MAX_FILES {
                return;
            }
        }
    }
}

/// True for the artifact extensions this explorer surfaces.
fn is_artifact(path: &Path) -> bool {
    matches!(
        ext_lower(path).as_str(),
        "json" | "jsonl" | "prismqnanl" | "tsv"
    )
}

/// Lowercased file extension, or an empty string when there is none.
fn ext_lower(path: &Path) -> String {
    path.extension()
        .map(|ext| ext.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default()
}

/// Build the right-pane preview text for `path`. Binary QNANL packets get a
/// note; text formats get their first [`MAX_PREVIEW_LINES`] lines. Never panics:
/// read failures and oversize files become inline notices.
fn load_preview(path: &Path) -> String {
    if ext_lower(path) == "prismqnanl" {
        return QNANL_NOTE.to_string();
    }

    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() > MAX_PREVIEW_BYTES {
            return format!(
                "File is {} — too large to preview inline.\n\n\
                 Only files up to {} are rendered here.",
                human_bytes(meta.len()),
                human_bytes(MAX_PREVIEW_BYTES),
            );
        }
    }

    match std::fs::read_to_string(path) {
        Ok(text) => {
            let total = text.lines().count();
            let shown: Vec<&str> = text.lines().take(MAX_PREVIEW_LINES).collect();
            let mut out = shown.join("\n");
            if total > MAX_PREVIEW_LINES {
                out.push_str(&format!(
                    "\n\n… truncated — showing the first {MAX_PREVIEW_LINES} of {total} lines."
                ));
            }
            if out.is_empty() {
                "(empty file)".to_string()
            } else {
                out
            }
        }
        Err(err) => format!("Could not read file:\n{err}"),
    }
}

/// Human-readable byte size, keeping the exact byte count for the audit trail.
fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    format!("{size:.1} {} ({bytes} bytes)", UNITS[unit])
}
