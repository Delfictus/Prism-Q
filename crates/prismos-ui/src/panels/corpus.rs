//! Tokenizer / corpus readiness panel.
//!
//! Loads `CORPUS_MERGE_READINESS.json` (the durable T4 corpus-merge readiness
//! object) and renders the per-artifact-family readiness table. The cardinal
//! doctrine of this surface is that readiness is judged by **expected matched
//! fields**, never by raw token count — so no token count is displayed
//! anywhere on this panel. Each row exposes its expected-matched-fields
//! expression verbatim plus a derived fidelity badge.

use adw::prelude::*;
use std::path::PathBuf;

use crate::app::AppContext;
use crate::model::corpus::{CorpusReadiness, CorpusRow};

/// Build the corpus readiness panel root widget.
pub fn build(ctx: &AppContext) -> gtk::Widget {
    // Candidate locations: the active installed run, then an explicit
    // deployment override. Missing evidence remains an empty, blocked state.
    let mut tried: Vec<PathBuf> = Vec::new();
    if let Some(run_dir) = ctx.run_dir() {
        tried.push(run_dir.join("CORPUS_MERGE_READINESS.json"));
    }
    if let Ok(configured) = std::env::var("PRISMOS_UI_CORPUS_READINESS") {
        tried.push(PathBuf::from(configured));
    }

    let chosen = tried.iter().find(|path| path.exists());

    match chosen {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) => match CorpusReadiness::from_json_str(&text) {
                Ok(doc) => build_loaded(path, &doc),
                Err(error) => {
                    empty_state(&tried, &format!("Failed to parse readiness JSON: {error}"))
                }
            },
            Err(error) => empty_state(&tried, &format!("Failed to read file: {error}")),
        },
        None => empty_state(
            &tried,
            "CORPUS_MERGE_READINESS.json not found at any candidate path.",
        ),
    }
}

/// Render the populated panel: header stats card + readiness table.
fn build_loaded(path: &std::path::Path, doc: &CorpusReadiness) -> gtk::Widget {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 14);
    root.set_margin_top(16);
    root.set_margin_bottom(16);
    root.set_margin_start(16);
    root.set_margin_end(16);

    root.append(&header_card(path, doc));

    // Table inside a scrolled window so long corpora scroll independently.
    let grid = gtk::Grid::new();
    grid.set_column_spacing(18);
    grid.set_row_spacing(8);
    grid.set_margin_top(4);
    grid.set_margin_bottom(4);
    grid.set_margin_start(4);
    grid.set_margin_end(4);

    let headers = [
        "Artifact family",
        "Tokenizer family",
        "Ready",
        "Expected matched_fields",
        "Fidelity",
        "Classification",
        "Blocker",
    ];
    for (col, title) in headers.iter().enumerate() {
        grid.attach(&header_cell(title), col as i32, 0, 1, 1);
    }

    if doc.rows.is_empty() {
        let empty = gtk::Label::new(Some("No artifact-family rows in this readiness object."));
        empty.add_css_class("dim");
        empty.set_xalign(0.0);
        grid.attach(&empty, 0, 1, headers.len() as i32, 1);
    } else {
        for (index, row) in doc.rows.iter().enumerate() {
            populate_row(&grid, (index + 1) as i32, row);
        }
    }

    let scroller = gtk::ScrolledWindow::builder()
        .hexpand(true)
        .vexpand(true)
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&grid)
        .build();
    scroller.add_css_class("card");
    root.append(&scroller);

    root.upcast()
}

/// The header card: overall verdict + labelled readiness stats + the doctrine
/// subtitle. No token count appears here by design.
fn header_card(path: &std::path::Path, doc: &CorpusReadiness) -> gtk::Widget {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 10);
    card.add_css_class("card");

    let title = gtk::Label::new(Some("Tokenizer / Corpus Readiness"));
    title.add_css_class("prismos-title");
    title.set_xalign(0.0);
    card.append(&title);

    let verdict_text = doc
        .overall_verdict()
        .unwrap_or_else(|| "unknown".to_string());
    let verdict_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let verdict_caption = gtk::Label::new(Some("Overall verdict:"));
    verdict_caption.add_css_class("dim");
    verdict_caption.set_xalign(0.0);
    let verdict_value = gtk::Label::new(Some(&verdict_text));
    verdict_value.add_css_class("mono");
    verdict_value.set_xalign(0.0);
    verdict_value.set_wrap(true);
    verdict_row.append(&verdict_caption);
    verdict_row.append(&verdict_value);
    card.append(&verdict_row);

    // Labelled stats.
    let stats = gtk::Box::new(gtk::Orientation::Horizontal, 28);
    stats.append(&stat(&doc.ready_now_count().to_string(), "Ready now"));
    stats.append(&stat(
        &doc.fidelity_expected_count().to_string(),
        "Fidelity expected",
    ));
    card.append(&stats);

    // The load-bearing doctrine line — verbatim.
    let doctrine = gtk::Label::new(Some(
        "Readiness keys off expected matched_fields, never token count.",
    ));
    doctrine.add_css_class("subtitle");
    doctrine.set_xalign(0.0);
    doctrine.set_wrap(true);
    card.append(&doctrine);

    // Provenance of the loaded file.
    let source = gtk::Label::new(Some(&path.display().to_string()));
    source.add_css_class("dim");
    source.set_xalign(0.0);
    source.set_wrap(true);
    card.append(&source);

    card.upcast()
}

/// Attach one corpus row's seven cells to `grid` at grid-row `r`.
fn populate_row(grid: &gtk::Grid, r: i32, row: &CorpusRow) {
    // Artifact family.
    grid.attach(&mono_cell(&row.artifact_family), 0, r, 1, 1);

    // Tokenizer family (or "-").
    let tokenizer = row.tokenizer_family.as_deref().unwrap_or("-");
    grid.attach(&text_cell(tokenizer), 1, r, 1, 1);

    // Ready pill.
    let (ready_text, ready_variant) = if row.tokenizer_family_ready == Some(true) {
        ("READY", "pass")
    } else {
        ("NOT READY", "blocked")
    };
    grid.attach(&status_pill(ready_text, ready_variant), 2, r, 1, 1);

    // Expected matched_fields — verbatim expression.
    grid.attach(&mono_cell(&row.matched_fields_display()), 3, r, 1, 1);

    // Fidelity badge.
    grid.attach(&fidelity_pill(row.fidelity_badge()), 4, r, 1, 1);

    // Classification.
    grid.attach(&text_cell(row.classification()), 5, r, 1, 1);

    // Blocker (dim, possibly empty).
    let blocker = row.blocker.as_deref().unwrap_or("");
    let blocker_label = gtk::Label::new(Some(blocker));
    blocker_label.add_css_class("dim");
    blocker_label.set_xalign(0.0);
    blocker_label.set_wrap(true);
    grid.attach(&blocker_label, 6, r, 1, 1);
}

/// A column header cell.
fn header_cell(text: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.add_css_class("subtitle");
    label.set_xalign(0.0);
    label
}

/// A plain left-aligned text cell.
fn text_cell(text: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.set_xalign(0.0);
    label.set_wrap(true);
    label
}

/// A monospaced left-aligned cell.
fn mono_cell(text: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.add_css_class("mono");
    label.set_xalign(0.0);
    label.set_wrap(true);
    label
}

/// A labelled statistic: a large value over a dim caption.
fn stat(value: &str, caption: &str) -> gtk::Widget {
    let column = gtk::Box::new(gtk::Orientation::Vertical, 2);
    let value_label = gtk::Label::new(Some(value));
    value_label.add_css_class("prismos-title");
    value_label.set_xalign(0.0);
    let caption_label = gtk::Label::new(Some(caption));
    caption_label.add_css_class("dim");
    caption_label.set_xalign(0.0);
    column.append(&value_label);
    column.append(&caption_label);
    column.upcast()
}

/// A status pill with the given `status-pill` variant class.
fn status_pill(text: &str, variant: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.add_css_class("status-pill");
    label.add_css_class(variant);
    label.set_halign(gtk::Align::Start);
    label
}

/// A fidelity pill derived from [`CorpusRow::fidelity_badge`]:
/// `FIDELITY_EXPECTED` -> info, `ZERO_FIELDS` -> warn, anything else -> dim.
fn fidelity_pill(badge: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(badge));
    label.set_halign(gtk::Align::Start);
    label.set_xalign(0.0);
    match badge {
        "FIDELITY_EXPECTED" => {
            label.add_css_class("status-pill");
            label.add_css_class("info");
        }
        "ZERO_FIELDS" => {
            label.add_css_class("status-pill");
            label.add_css_class("warn");
        }
        _ => {
            label.add_css_class("dim");
        }
    }
    label
}

/// Empty-state card shown when the readiness file is missing or unparseable.
/// Shows every candidate path that was tried plus the error.
fn empty_state(tried: &[PathBuf], error: &str) -> gtk::Widget {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 14);
    root.set_margin_top(16);
    root.set_margin_bottom(16);
    root.set_margin_start(16);
    root.set_margin_end(16);

    let card = gtk::Box::new(gtk::Orientation::Vertical, 10);
    card.add_css_class("card");

    let title = gtk::Label::new(Some("Corpus readiness unavailable"));
    title.add_css_class("prismos-title");
    title.set_xalign(0.0);
    card.append(&title);

    let error_label = gtk::Label::new(Some(error));
    error_label.add_css_class("dim");
    error_label.set_xalign(0.0);
    error_label.set_wrap(true);
    card.append(&error_label);

    let doctrine = gtk::Label::new(Some(
        "Readiness keys off expected matched_fields, never token count.",
    ));
    doctrine.add_css_class("subtitle");
    doctrine.set_xalign(0.0);
    doctrine.set_wrap(true);
    card.append(&doctrine);

    let tried_title = gtk::Label::new(Some("Paths tried:"));
    tried_title.add_css_class("subtitle");
    tried_title.set_xalign(0.0);
    card.append(&tried_title);

    for path in tried {
        let path_label = gtk::Label::new(Some(&path.display().to_string()));
        path_label.add_css_class("mono");
        path_label.set_xalign(0.0);
        path_label.set_wrap(true);
        card.append(&path_label);
    }

    root.append(&card);
    root.upcast()
}
