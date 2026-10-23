//! Cryptic-site analysis panel.
//!
//! This is the direct UI surface for the `prism-cryptic-site-analysis`
//! backend. It ranks cryptic Path-B sites against the matching orthosteric
//! branch and shows the exact residue/centroid evidence needed to decide which
//! candidates deserve follow-up.

use std::path::{Path, PathBuf};

use adw::prelude::*;
use gtk::{gio, glib};
use serde_json::Value;

use crate::adapters::PrismCommand;
use crate::app::AppContext;
use crate::jobs::JobEvent;

#[derive(Debug, Clone)]
struct CrypticSummary {
    run_root: String,
    candidate_count: u64,
    high_count: u64,
    moderate_count: u64,
    orthosteric_match_count: u64,
    rows: Vec<CrypticRow>,
}

#[derive(Debug, Clone)]
struct CrypticRow {
    branch: String,
    site_id: String,
    rank: i64,
    classification: String,
    candidate_score: Option<f64>,
    novelty_score: Option<f64>,
    support_score: Option<f64>,
    nearest_orthosteric_site: String,
    orthosteric_distance_a: Option<f64>,
    orthosteric_union_jaccard: Option<f64>,
    cross_ligand_nearest_site: String,
    cross_ligand_distance_a: Option<f64>,
    cross_ligand_union_jaccard: Option<f64>,
    known_reference_overlap_count: u64,
    known_reference_overlap_residues: String,
    representative_residues: String,
    lining_residues: String,
    centroid_xyz: String,
    stream_count: Option<u64>,
    spike_count: Option<u64>,
}

/// Build the panel and refresh on every map, so a newly selected run root or
/// a just-finished analysis is reflected without restarting the app.
pub fn build(ctx: &AppContext) -> gtk::Widget {
    let container = gtk::Box::new(gtk::Orientation::Vertical, 0);
    container.set_hexpand(true);
    container.set_vexpand(true);
    let ctx_map = ctx.clone();
    container.connect_map(move |c| populate(c, &ctx_map));
    container.upcast()
}

fn populate(container: &gtk::Box, ctx: &AppContext) {
    while let Some(child) = container.first_child() {
        container.remove(&child);
    }
    container.append(&build_body(ctx, container));
}

fn build_body(ctx: &AppContext, refresh_container: &gtk::Box) -> gtk::Widget {
    let Some(run_dir) = ctx.run_dir().filter(|path| path.is_dir()) else {
        return empty_state(
            "Cryptic-site analysis unavailable",
            "No run root is selected. Open Run Root and select a PrismOS native run.",
        );
    };

    let out_dir = default_output_dir();
    let summary = load_summary(&out_dir);

    let root = gtk::Box::new(gtk::Orientation::Vertical, 14);
    root.set_margin_top(16);
    root.set_margin_bottom(16);
    root.set_margin_start(16);
    root.set_margin_end(16);

    root.append(&header_card(&run_dir, summary.as_ref(), &out_dir));
    root.append(&action_bar(
        ctx,
        &run_dir,
        &out_dir,
        summary.as_ref(),
        refresh_container,
    ));

    if let Some(summary) = summary {
        root.append(&candidate_table(&summary.rows));
        root.append(&readout_card());
    } else {
        root.append(&empty_results_card(&out_dir));
    }

    let scroller = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .hexpand(true)
        .vexpand(true)
        .child(&root)
        .build();
    scroller.upcast()
}

fn header_card(run_dir: &Path, summary: Option<&CrypticSummary>, out_dir: &Path) -> gtk::Widget {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 10);
    card.add_css_class("card");

    let title = gtk::Label::new(Some("Cryptic Sites"));
    title.add_css_class("prismos-title");
    title.set_xalign(0.0);
    card.append(&title);

    let stats = gtk::Box::new(gtk::Orientation::Horizontal, 28);
    stats.append(&stat(
        &summary
            .map(|s| s.candidate_count.to_string())
            .unwrap_or_else(|| "-".to_string()),
        "Rows scored",
    ));
    stats.append(&stat(
        &summary
            .map(|s| s.high_count.to_string())
            .unwrap_or_else(|| "-".to_string()),
        "High priority",
    ));
    stats.append(&stat(
        &summary
            .map(|s| s.moderate_count.to_string())
            .unwrap_or_else(|| "-".to_string()),
        "Moderate priority",
    ));
    stats.append(&stat(
        &summary
            .map(|s| s.orthosteric_match_count.to_string())
            .unwrap_or_else(|| "-".to_string()),
        "Orthosteric matches",
    ));
    card.append(&stats);

    let source = gtk::Label::new(Some(&format!("run root: {}", run_dir.display())));
    source.add_css_class("mono");
    source.add_css_class("dim");
    source.set_xalign(0.0);
    source.set_wrap(true);
    card.append(&source);

    let output = gtk::Label::new(Some(&format!("analysis output: {}", out_dir.display())));
    output.add_css_class("mono");
    output.add_css_class("dim");
    output.set_xalign(0.0);
    output.set_wrap(true);
    card.append(&output);

    if let Some(summary) = summary {
        let bound_run = gtk::Label::new(Some(&format!("analysis run root: {}", summary.run_root)));
        bound_run.add_css_class("mono");
        bound_run.add_css_class("dim");
        bound_run.set_xalign(0.0);
        bound_run.set_wrap(true);
        card.append(&bound_run);
    }

    card.upcast()
}

fn action_bar(
    ctx: &AppContext,
    run_dir: &Path,
    out_dir: &Path,
    summary: Option<&CrypticSummary>,
    refresh_container: &gtk::Box,
) -> gtk::Widget {
    let column = gtk::Box::new(gtk::Orientation::Vertical, 8);
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    row.set_halign(gtk::Align::Start);

    let run_btn = gtk::Button::with_label("Run Analysis");
    run_btn.add_css_class("accent-button");
    let refresh_btn = gtk::Button::with_label("Refresh");
    let open_btn = gtk::Button::with_label("Open Report");
    let copy_btn = gtk::Button::with_label("Copy TSV");
    let export_btn = gtk::Button::with_label("Export TSV");

    let transcript = gtk::TextView::new();
    transcript.set_editable(false);
    transcript.set_cursor_visible(false);
    transcript.set_monospace(true);
    transcript.set_wrap_mode(gtk::WrapMode::None);
    transcript.add_css_class("mono");
    transcript.add_css_class("log-view");
    transcript
        .buffer()
        .set_text("Run the analysis or open the current report.\n");

    {
        let ctx = ctx.clone();
        let run_dir = run_dir.to_path_buf();
        let out_dir = out_dir.to_path_buf();
        let actions = row.clone();
        let buffer = transcript.buffer();
        let refresh_container = refresh_container.clone();
        run_btn.connect_clicked(move |_| {
            actions.set_sensitive(false);
            buffer.set_text("");
            let cmd = PrismCommand::cryptic_site_analysis(&run_dir, &out_dir);
            let argv = cmd.display(&ctx.resolver);
            append_line(&buffer, &format!("$ {argv}"));
            let buf_events = buffer.clone();
            let ctx_events = ctx.clone();
            let actions_done = actions.clone();
            let refresh = refresh_container.clone();
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
                    actions_done.set_sensitive(true);
                    if outcome.success {
                        ctx_events.toast("Cryptic-site analysis completed");
                        populate(&refresh, &ctx_events);
                    } else {
                        ctx_events.toast("Cryptic-site analysis failed");
                    }
                }
            });
        });
    }

    {
        let ctx = ctx.clone();
        let refresh_container = refresh_container.clone();
        refresh_btn.connect_clicked(move |_| {
            populate(&refresh_container, &ctx);
            ctx.toast("Cryptic-site panel refreshed");
        });
    }

    {
        let ctx = ctx.clone();
        let report = out_dir.join("cryptic_site_analysis.html");
        open_btn.connect_clicked(move |_| {
            if report.is_file() {
                match open_file(&report) {
                    Ok(()) => ctx.toast("Opened cryptic-site report"),
                    Err(err) => ctx.toast(&format!("Open failed: {err}")),
                }
            } else {
                ctx.toast("Run analysis first; report file does not exist");
            }
        });
    }

    let tsv = summary
        .map(|s| candidate_tsv(&s.rows))
        .unwrap_or_else(|| String::from("No cryptic-site analysis loaded.\n"));
    {
        let ctx = ctx.clone();
        let tsv = tsv.clone();
        copy_btn.connect_clicked(move |_| {
            copy_text(&tsv);
            ctx.toast("Cryptic-site rows copied as TSV");
        });
    }
    {
        let ctx = ctx.clone();
        let tsv = tsv.clone();
        export_btn.connect_clicked(move |_| match export_text("cryptic_sites.tsv", &tsv) {
            Ok(path) => ctx.toast(&format!("Exported cryptic-site rows: {}", path.display())),
            Err(err) => ctx.toast(&format!("Export failed: {err}")),
        });
    }

    row.append(&run_btn);
    row.append(&refresh_btn);
    row.append(&open_btn);
    row.append(&copy_btn);
    row.append(&export_btn);
    column.append(&row);

    let scroller = gtk::ScrolledWindow::builder()
        .min_content_height(96)
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&transcript)
        .build();
    scroller.add_css_class("card");
    column.append(&scroller);
    column.upcast()
}

fn candidate_table(rows: &[CrypticRow]) -> gtk::Widget {
    let grid = gtk::Grid::new();
    grid.set_column_spacing(16);
    grid.set_row_spacing(8);
    grid.set_margin_top(10);
    grid.set_margin_bottom(10);
    grid.set_margin_start(10);
    grid.set_margin_end(10);

    let headers = [
        "Branch",
        "Site",
        "Rank",
        "Class",
        "Candidate",
        "Novelty",
        "Support",
        "Nearest orthosteric",
        "Distance A",
        "Residue overlap",
        "Cross-ligand site",
        "Cross distance A",
        "Cross overlap",
        "Known overlap",
        "Representative",
        "Lining residues",
        "Centroid",
        "Streams",
        "Spikes",
    ];
    for (col, title) in headers.iter().enumerate() {
        grid.attach(&header_cell(title), col as i32, 0, 1, 1);
    }

    for (idx, row) in rows.iter().enumerate() {
        let r = (idx + 1) as i32;
        grid.attach(&mono_cell(&row.branch), 0, r, 1, 1);
        grid.attach(&mono_cell(&row.site_id), 1, r, 1, 1);
        grid.attach(&mono_cell(&row.rank.to_string()), 2, r, 1, 1);
        grid.attach(&classification_pill(&row.classification), 3, r, 1, 1);
        grid.attach(&mono_cell(&opt_f64(row.candidate_score)), 4, r, 1, 1);
        grid.attach(&mono_cell(&opt_f64(row.novelty_score)), 5, r, 1, 1);
        grid.attach(&mono_cell(&opt_f64(row.support_score)), 6, r, 1, 1);
        grid.attach(&mono_cell(&row.nearest_orthosteric_site), 7, r, 1, 1);
        grid.attach(&mono_cell(&opt_f64(row.orthosteric_distance_a)), 8, r, 1, 1);
        grid.attach(
            &mono_cell(&opt_f64(row.orthosteric_union_jaccard)),
            9,
            r,
            1,
            1,
        );
        grid.attach(&mono_cell(&row.cross_ligand_nearest_site), 10, r, 1, 1);
        grid.attach(
            &mono_cell(&opt_f64(row.cross_ligand_distance_a)),
            11,
            r,
            1,
            1,
        );
        grid.attach(
            &mono_cell(&opt_f64(row.cross_ligand_union_jaccard)),
            12,
            r,
            1,
            1,
        );
        grid.attach(
            &mono_cell(&format!(
                "{} ({})",
                row.known_reference_overlap_count, row.known_reference_overlap_residues
            )),
            13,
            r,
            1,
            1,
        );
        grid.attach(&mono_cell(&row.representative_residues), 14, r, 1, 1);
        grid.attach(&mono_cell(&row.lining_residues), 15, r, 1, 1);
        grid.attach(&mono_cell(&row.centroid_xyz), 16, r, 1, 1);
        grid.attach(&mono_cell(&opt_u64(row.stream_count)), 17, r, 1, 1);
        grid.attach(&mono_cell(&opt_u64(row.spike_count)), 18, r, 1, 1);
    }

    let scroller = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .hexpand(true)
        .vexpand(true)
        .child(&grid)
        .build();
    scroller.add_css_class("card");
    scroller.upcast()
}

fn readout_card() -> gtk::Widget {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 8);
    card.add_css_class("card");
    let title = gtk::Label::new(Some("Readout"));
    title.add_css_class("subtitle");
    title.set_xalign(0.0);
    card.append(&title);
    for line in [
        "High-priority rows are cryptic-specific by centroid/residue separation and supported by stream, spike, KCC, temporal, and morphology signals.",
        "Rows marked as orthosteric matches are useful controls, not cryptic-site hits.",
        "Known-overlap rows intersect the local 4LPK/GDP switch residue set and should be reviewed separately from novel cryptic candidates.",
    ] {
        let label = gtk::Label::new(Some(line));
        label.set_xalign(0.0);
        label.set_wrap(true);
        card.append(&label);
    }
    card.upcast()
}

fn empty_results_card(out_dir: &Path) -> gtk::Widget {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 8);
    card.add_css_class("card");
    let title = gtk::Label::new(Some("No analysis loaded"));
    title.add_css_class("subtitle");
    title.set_xalign(0.0);
    card.append(&title);
    let body = gtk::Label::new(Some(&format!(
        "Click Run Analysis. The report and candidate table will be written to {}.",
        out_dir.display()
    )));
    body.set_xalign(0.0);
    body.set_wrap(true);
    card.append(&body);
    card.upcast()
}

fn load_summary(out_dir: &Path) -> Option<CrypticSummary> {
    let doc = json_at(&out_dir.join("cryptic_site_analysis.json"))?;
    let run_root = doc.get("run_root")?.as_str()?.to_string();
    let counts = doc.get("classification_counts")?;
    let rows = doc
        .get("top_candidates")
        .and_then(Value::as_array)
        .map(|vals| vals.iter().filter_map(parse_row).collect::<Vec<_>>())
        .unwrap_or_default();
    Some(CrypticSummary {
        run_root,
        candidate_count: doc
            .get("candidate_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        high_count: counts
            .get("HIGH_PRIORITY_CRYPTIC_SITE_CANDIDATE")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        moderate_count: counts
            .get("MODERATE_PRIORITY_CRYPTIC_SITE_CANDIDATE")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        orthosteric_match_count: counts
            .get("MATCHES_ORTHOSTERIC_SITE")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        rows,
    })
}

fn parse_row(value: &Value) -> Option<CrypticRow> {
    Some(CrypticRow {
        branch: str_value(value, "branch")?,
        site_id: str_value(value, "site_id")?,
        rank: value.get("rank").and_then(Value::as_i64).unwrap_or(0),
        classification: str_value(value, "classification")?,
        candidate_score: num_value(value, "cryptic_candidate_score"),
        novelty_score: num_value(value, "novelty_score"),
        support_score: num_value(value, "support_score"),
        nearest_orthosteric_site: str_value(value, "nearest_orthosteric_site").unwrap_or_default(),
        orthosteric_distance_a: num_value(value, "orthosteric_centroid_distance_a"),
        orthosteric_union_jaccard: num_value(value, "orthosteric_union_jaccard"),
        cross_ligand_nearest_site: str_value(value, "cross_ligand_nearest_site")
            .unwrap_or_default(),
        cross_ligand_distance_a: num_value(value, "cross_ligand_centroid_distance_a"),
        cross_ligand_union_jaccard: num_value(value, "cross_ligand_union_jaccard"),
        known_reference_overlap_count: value
            .get("known_reference_overlap_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        known_reference_overlap_residues: str_value(value, "known_reference_overlap_residues")
            .unwrap_or_else(|| "-".to_string()),
        representative_residues: str_value(value, "representative_residues")
            .unwrap_or_else(|| "-".to_string()),
        lining_residues: str_value(value, "lining_residues").unwrap_or_else(|| "-".to_string()),
        centroid_xyz: str_value(value, "centroid_xyz").unwrap_or_else(|| "-".to_string()),
        stream_count: value.get("stream_count").and_then(Value::as_u64),
        spike_count: value.get("spike_count").and_then(Value::as_u64),
    })
}

fn default_output_dir() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("PrismOS-Latest-Results")
        .join("cryptic_site_analysis")
}

fn json_at(path: &Path) -> Option<Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
}

fn str_value(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

fn num_value(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

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

fn header_cell(text: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.add_css_class("subtitle");
    label.set_xalign(0.0);
    label
}

fn mono_cell(text: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.add_css_class("mono");
    label.set_xalign(0.0);
    label.set_wrap(false);
    label
}

fn classification_pill(text: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.add_css_class("status-pill");
    let pass = text == "HIGH_PRIORITY_CRYPTIC_SITE_CANDIDATE"
        || text == "MODERATE_PRIORITY_CRYPTIC_SITE_CANDIDATE";
    label.add_css_class(if pass { "pass" } else { "warn" });
    label.set_halign(gtk::Align::Start);
    label
}

fn opt_f64(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "-".to_string())
}

fn opt_u64(value: Option<u64>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn candidate_tsv(rows: &[CrypticRow]) -> String {
    let mut out = String::from(
        "branch\tsite_id\trank\tclassification\tcandidate_score\tnovelty_score\tsupport_score\tnearest_orthosteric_site\torthosteric_distance_a\torthosteric_union_jaccard\tcross_ligand_nearest_site\tcross_ligand_distance_a\tcross_ligand_union_jaccard\tknown_reference_overlap_count\tknown_reference_overlap_residues\trepresentative_residues\tlining_residues\tcentroid_xyz\tstream_count\tspike_count\n",
    );
    for row in rows {
        let cells = [
            row.branch.clone(),
            row.site_id.clone(),
            row.rank.to_string(),
            row.classification.clone(),
            opt_f64(row.candidate_score),
            opt_f64(row.novelty_score),
            opt_f64(row.support_score),
            row.nearest_orthosteric_site.clone(),
            opt_f64(row.orthosteric_distance_a),
            opt_f64(row.orthosteric_union_jaccard),
            row.cross_ligand_nearest_site.clone(),
            opt_f64(row.cross_ligand_distance_a),
            opt_f64(row.cross_ligand_union_jaccard),
            row.known_reference_overlap_count.to_string(),
            row.known_reference_overlap_residues.clone(),
            row.representative_residues.clone(),
            row.lining_residues.clone(),
            row.centroid_xyz.clone(),
            opt_u64(row.stream_count),
            opt_u64(row.spike_count),
        ];
        out.push_str(
            &cells
                .iter()
                .map(|cell| tsv_cell(cell))
                .collect::<Vec<_>>()
                .join("\t"),
        );
        out.push('\n');
    }
    out
}

fn copy_text(text: &str) {
    if let Some(display) = gtk::gdk::Display::default() {
        display.clipboard().set_text(text);
    }
}

fn export_text(filename: &str, text: &str) -> std::io::Result<PathBuf> {
    let base = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("PrismOS-Latest-Results")
        .join("exports");
    std::fs::create_dir_all(&base)?;
    let path = base.join(filename);
    std::fs::write(&path, text)?;
    Ok(path)
}

fn open_file(path: &Path) -> Result<(), glib::Error> {
    let uri = glib::filename_to_uri(path, None::<&str>)?;
    gio::AppInfo::launch_default_for_uri(&uri, None::<&gio::AppLaunchContext>)
}

fn append_line(buffer: &gtk::TextBuffer, text: &str) {
    let mut end = buffer.end_iter();
    buffer.insert(&mut end, text);
    buffer.insert(&mut end, "\n");
}

fn empty_state(title: &str, description: &str) -> gtk::Widget {
    adw::StatusPage::builder()
        .title(title)
        .description(description)
        .icon_name("dialog-information-symbolic")
        .build()
        .upcast()
}

fn tsv_cell(value: &str) -> String {
    value
        .replace('\t', " ")
        .replace('\n', " ")
        .replace('\r', " ")
}
