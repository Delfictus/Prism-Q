//! Scientific-results panel.
//!
//! This panel summarizes the science-bearing outputs from the active PrismOS
//! run. It intentionally separates scientific signals from provenance signals:
//! receipts, QNANL packets, hashes, and logs belong in Evidence/QNANL unless
//! they also carry a scientific metric.

use std::path::Path;

use adw::prelude::*;
use serde_json::Value;

use crate::app::AppContext;
use crate::model::run_layout;

#[derive(Debug, Clone)]
struct ScienceBranch {
    branch: String,
    detection_mode: String,
    ligand_mode: String,
    gdp_present: String,
    binding_sites: usize,
    partial_sites: usize,
    top_site: String,
    top_residues: String,
    top_family_rank: Option<f64>,
    top_morphology_rank: Option<f64>,
    top_physiology_rank: Option<f64>,
    top_kcc_score: Option<f64>,
    twin_cells: usize,
    twin_pairwise_residuals: Option<u64>,
    twin_factorial_status: String,
    twin_spikes: Option<u64>,
    thermo_cells: usize,
    bridge_groups: Option<u64>,
    tensor_summaries: usize,
    tensor_summary_present: usize,
    tokenizer_matched_fields: Option<u64>,
    tokenizer_records: Option<u64>,
    tokenizer_gate: String,
}

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
    container.append(&build_body(ctx));
}

fn build_body(ctx: &AppContext) -> gtk::Widget {
    let Some(run_dir) = ctx.run_dir().filter(|path| path.is_dir()) else {
        return empty_state(
            "Scientific results unavailable",
            "No run root is selected. Open Run Root and select a PrismOS native run.",
        );
    };

    let rows = load_scientific_rows(&run_dir);
    if rows.is_empty() {
        return empty_state(
            "Scientific results unavailable",
            "No native modality branches were found under the selected run root.",
        );
    }

    let root = gtk::Box::new(gtk::Orientation::Vertical, 14);
    root.set_margin_top(16);
    root.set_margin_bottom(16);
    root.set_margin_start(16);
    root.set_margin_end(16);

    root.append(&header_card(&run_dir, &rows));
    root.append(&action_bar(ctx, &rows));
    root.append(&branch_table(&rows));
    root.append(&surface_table(&rows));
    root.append(&next_readout_card());

    let scroller = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .hexpand(true)
        .vexpand(true)
        .child(&root)
        .build();
    scroller.upcast()
}

fn load_scientific_rows(run_dir: &Path) -> Vec<ScienceBranch> {
    let mut rows = Vec::new();
    // Discover native modality branches from the on-disk evidence rather than a
    // hardcoded name list: any directory carrying a science-bearing modality
    // artifact (modality receipt, TWIN residuals, thermo ledger, tensor
    // summaries, tokenizer receipt, or a materialized Path-B stage).
    for branch_info in run_layout::discover_modality_branches(run_dir) {
        let branch_dir = branch_info.path.clone();
        rows.push(ScienceBranch {
            branch: branch_info.name.clone(),
            detection_mode: modality_value(&branch_dir, "detection_mode"),
            ligand_mode: modality_value(&branch_dir, "ligand_mode"),
            gdp_present: gdp_present(&branch_dir),
            binding_sites: binding_site_count(&branch_dir),
            partial_sites: partial_site_count(&branch_dir),
            top_site: top_site_value(&branch_dir, "site_id"),
            top_residues: top_site_residues(&branch_dir),
            top_family_rank: top_site_number(
                &branch_dir,
                &["lexicographic_rank_key", "site_family_preservation_rank"],
            )
            .or_else(|| {
                top_site_number(
                    &branch_dir,
                    &["site_family_preservation", "site_family_preservation_rank"],
                )
            }),
            top_morphology_rank: top_site_number(
                &branch_dir,
                &["lexicographic_rank_key", "pocket_morphology_rank"],
            )
            .or_else(|| {
                top_site_number(
                    &branch_dir,
                    &["pocket_morphology_surface", "morphology_rank"],
                )
            }),
            top_physiology_rank: top_site_number(
                &branch_dir,
                &["lexicographic_rank_key", "thermodynamic_physiology_rank"],
            )
            .or_else(|| {
                top_site_number(
                    &branch_dir,
                    &["binding_site_physiology_surface", "physiology_rank"],
                )
            }),
            top_kcc_score: top_site_number(&branch_dir, &["kcc_driver", "kcc_driver_score"]),
            twin_cells: twin_cells(&branch_dir),
            twin_pairwise_residuals: json_at(
                &branch_dir.join("TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS.json"),
            )
            .and_then(|v| v.get("pairwise_residual_count").and_then(Value::as_u64)),
            twin_factorial_status: json_at(
                &branch_dir.join("TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS.json"),
            )
            .and_then(|v| {
                v.get("factorial_status")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "-".to_string()),
            twin_spikes: twin_spike_total(&branch_dir),
            thermo_cells: thermo_cells(&branch_dir),
            bridge_groups: json_at(
                &branch_dir.join("thermodynamic_phase_2x2x2x2_condition_ledger.json"),
            )
            .and_then(|v| v.get("bridge_group_count").and_then(Value::as_u64)),
            tensor_summaries: tensor_count(&branch_dir).0,
            tensor_summary_present: tensor_count(&branch_dir).1,
            tokenizer_matched_fields: json_at(
                &branch_dir.join("prismos_runtime_tokenizer_export_receipt.json"),
            )
            .and_then(|v| v.get("n_matched_fields").and_then(Value::as_u64)),
            tokenizer_records: json_at(
                &branch_dir.join("prismos_runtime_tokenizer_export_receipt.json"),
            )
            .and_then(|v| v.get("n_records").and_then(Value::as_u64)),
            tokenizer_gate: json_at(
                &branch_dir.join("prismos_runtime_tokenizer_export_receipt.json"),
            )
            .and_then(|v| {
                v.get("matched_fields_gate")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "-".to_string()),
        });
    }
    rows
}

fn header_card(run_dir: &Path, rows: &[ScienceBranch]) -> gtk::Widget {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 10);
    card.add_css_class("card");

    let title = gtk::Label::new(Some("Scientific Results"));
    title.add_css_class("prismos-title");
    title.set_xalign(0.0);
    card.append(&title);

    let total_sites: usize = rows.iter().map(|r| r.binding_sites).sum();
    let total_spikes: u64 = rows.iter().filter_map(|r| r.twin_spikes).sum();
    let total_matched: u64 = rows.iter().filter_map(|r| r.tokenizer_matched_fields).sum();

    let stats = gtk::Box::new(gtk::Orientation::Horizontal, 28);
    stats.append(&stat(&rows.len().to_string(), "Branches"));
    stats.append(&stat(&total_sites.to_string(), "Binding sites"));
    stats.append(&stat(&total_spikes.to_string(), "TWIN spike support"));
    stats.append(&stat(&total_matched.to_string(), "Matched fields"));
    card.append(&stats);

    let subtitle = gtk::Label::new(Some(
        "Science-bearing metrics only. Receipts, hashes, QNANL packets, and logs stay in Evidence/QNANL.",
    ));
    subtitle.add_css_class("subtitle");
    subtitle.set_xalign(0.0);
    subtitle.set_wrap(true);
    card.append(&subtitle);

    let source = gtk::Label::new(Some(&format!("run root: {}", run_dir.display())));
    source.add_css_class("mono");
    source.add_css_class("dim");
    source.set_xalign(0.0);
    source.set_wrap(true);
    card.append(&source);

    card.upcast()
}

fn action_bar(ctx: &AppContext, rows: &[ScienceBranch]) -> gtk::Widget {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    row.set_halign(gtk::Align::Start);

    let copy_btn = gtk::Button::with_label("Copy Summary");
    copy_btn.add_css_class("accent-button");
    let export_btn = gtk::Button::with_label("Export Summary");

    let tsv = science_tsv(rows);
    {
        let ctx = ctx.clone();
        let tsv = tsv.clone();
        copy_btn.connect_clicked(move |_| {
            copy_text(&tsv);
            ctx.toast("Scientific results copied as TSV");
        });
    }
    {
        let ctx = ctx.clone();
        let tsv = tsv.clone();
        export_btn.connect_clicked(move |_| match export_text("scientific_results.tsv", &tsv) {
            Ok(path) => ctx.toast(&format!("Exported scientific results: {}", path.display())),
            Err(err) => ctx.toast(&format!("Export failed: {err}")),
        });
    }

    row.append(&copy_btn);
    row.append(&export_btn);
    row.upcast()
}

fn branch_table(rows: &[ScienceBranch]) -> gtk::Widget {
    let grid = gtk::Grid::new();
    grid.set_column_spacing(16);
    grid.set_row_spacing(8);
    grid.set_margin_top(10);
    grid.set_margin_bottom(10);
    grid.set_margin_start(10);
    grid.set_margin_end(10);
    let headers = [
        "Branch",
        "Detection",
        "Ligand",
        "GDP",
        "Sites",
        "Top site",
        "Top residues",
        "Family",
        "Morphology",
        "Physiology",
        "KCC",
    ];
    for (col, title) in headers.iter().enumerate() {
        grid.attach(&header_cell(title), col as i32, 0, 1, 1);
    }
    for (idx, row) in rows.iter().enumerate() {
        let r = (idx + 1) as i32;
        grid.attach(&mono_cell(&row.branch), 0, r, 1, 1);
        grid.attach(&text_cell(&row.detection_mode), 1, r, 1, 1);
        grid.attach(&text_cell(&row.ligand_mode), 2, r, 1, 1);
        grid.attach(&text_cell(&row.gdp_present), 3, r, 1, 1);
        grid.attach(&mono_cell(&row.binding_sites.to_string()), 4, r, 1, 1);
        grid.attach(&mono_cell(&row.top_site), 5, r, 1, 1);
        grid.attach(&mono_cell(&row.top_residues), 6, r, 1, 1);
        grid.attach(&mono_cell(&fmt_f64(row.top_family_rank)), 7, r, 1, 1);
        grid.attach(&mono_cell(&fmt_f64(row.top_morphology_rank)), 8, r, 1, 1);
        grid.attach(&mono_cell(&fmt_f64(row.top_physiology_rank)), 9, r, 1, 1);
        grid.attach(&mono_cell(&fmt_f64(row.top_kcc_score)), 10, r, 1, 1);
    }
    scroll_card(grid, 210)
}

fn surface_table(rows: &[ScienceBranch]) -> gtk::Widget {
    let grid = gtk::Grid::new();
    grid.set_column_spacing(16);
    grid.set_row_spacing(8);
    grid.set_margin_top(10);
    grid.set_margin_bottom(10);
    grid.set_margin_start(10);
    grid.set_margin_end(10);
    let headers = [
        "Branch",
        "TWIN cells",
        "Pair residuals",
        "Factorial status",
        "Spike support",
        "Thermo cells",
        "Bridge groups",
        "Tensor summaries",
        "Tokenizer rows",
        "Matched fields",
        "Gate",
    ];
    for (col, title) in headers.iter().enumerate() {
        grid.attach(&header_cell(title), col as i32, 0, 1, 1);
    }
    for (idx, row) in rows.iter().enumerate() {
        let r = (idx + 1) as i32;
        grid.attach(&mono_cell(&row.branch), 0, r, 1, 1);
        grid.attach(&mono_cell(&row.twin_cells.to_string()), 1, r, 1, 1);
        grid.attach(
            &mono_cell(&fmt_u64(row.twin_pairwise_residuals)),
            2,
            r,
            1,
            1,
        );
        grid.attach(&text_cell(&row.twin_factorial_status), 3, r, 1, 1);
        grid.attach(&mono_cell(&fmt_u64(row.twin_spikes)), 4, r, 1, 1);
        grid.attach(&mono_cell(&row.thermo_cells.to_string()), 5, r, 1, 1);
        grid.attach(&mono_cell(&fmt_u64(row.bridge_groups)), 6, r, 1, 1);
        grid.attach(
            &mono_cell(&format!(
                "{}/{}",
                row.tensor_summary_present, row.tensor_summaries
            )),
            7,
            r,
            1,
            1,
        );
        grid.attach(&mono_cell(&fmt_u64(row.tokenizer_records)), 8, r, 1, 1);
        grid.attach(
            &mono_cell(&fmt_u64(row.tokenizer_matched_fields)),
            9,
            r,
            1,
            1,
        );
        grid.attach(
            &status_pill(&row.tokenizer_gate, row.tokenizer_gate == "PASS"),
            10,
            r,
            1,
            1,
        );
    }
    scroll_card(grid, 210)
}

fn next_readout_card() -> gtk::Widget {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 8);
    card.add_css_class("card");
    let title = gtk::Label::new(Some("Readout order"));
    title.add_css_class("subtitle");
    title.set_xalign(0.0);
    card.append(&title);
    for line in [
        "1. Within each ligand state, compare orthosteric vs cryptic detection-mode branches for detection-mode effects.",
        "2. Compare each holo branch against the matching apo branch for ligand/GDP-state effects.",
        "3. Open Binding Sites for residue-level site rows and scores.",
        "4. Open State Graph for state transitions; open Evidence/QNANL only when provenance or durability proof is needed.",
    ] {
        let label = gtk::Label::new(Some(line));
        label.set_xalign(0.0);
        label.set_wrap(true);
        card.append(&label);
    }
    card.upcast()
}

fn scroll_card(grid: gtk::Grid, min_height: i32) -> gtk::Widget {
    let scroller = gtk::ScrolledWindow::builder()
        .min_content_height(min_height)
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .hexpand(true)
        .child(&grid)
        .build();
    scroller.add_css_class("card");
    scroller.upcast()
}

fn json_at(path: &Path) -> Option<Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
}

fn modality_value(branch_dir: &Path, key: &str) -> String {
    json_at(&branch_dir.join("prismos_modality_receipt.json"))
        .and_then(|v| v.get(key).and_then(Value::as_str).map(str::to_string))
        .unwrap_or_else(|| "-".to_string())
}

fn gdp_present(branch_dir: &Path) -> String {
    json_at(&branch_dir.join("prismos_modality_receipt.json"))
        .and_then(|v| v.get("gdp_present_in_topology").and_then(Value::as_bool))
        .map(|v| if v { "present" } else { "absent" }.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn binding_doc(branch_dir: &Path) -> Option<Value> {
    json_at(&branch_dir.join("pathb_runtime_stage/binding_sites.materialized.json"))
}

fn binding_site_count(branch_dir: &Path) -> usize {
    binding_doc(branch_dir)
        .and_then(|v| {
            v.get("binding_sites")
                .and_then(Value::as_array)
                .map(Vec::len)
        })
        .unwrap_or(0)
}

fn partial_site_count(branch_dir: &Path) -> usize {
    binding_doc(branch_dir)
        .and_then(|v| {
            v.get("spatiotemporal_partial_candidates")
                .and_then(Value::as_array)
                .map(Vec::len)
        })
        .unwrap_or(0)
}

fn top_site(branch_dir: &Path) -> Option<Value> {
    binding_doc(branch_dir).and_then(|v| {
        v.get("binding_sites")
            .and_then(Value::as_array)
            .and_then(|rows| rows.first().cloned())
    })
}

fn top_site_value(branch_dir: &Path, key: &str) -> String {
    top_site(branch_dir)
        .and_then(|v| v.get(key).and_then(Value::as_str).map(str::to_string))
        .unwrap_or_else(|| "-".to_string())
}

fn top_site_residues(branch_dir: &Path) -> String {
    let Some(site) = top_site(branch_dir) else {
        return "-".to_string();
    };
    format_int_list(site.get("representative_residues"), 8)
}

fn top_site_number(branch_dir: &Path, path: &[&str]) -> Option<f64> {
    let site = top_site(branch_dir)?;
    number_path(&site, path)
}

fn twin_cells(branch_dir: &Path) -> usize {
    json_at(&branch_dir.join("TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS.json"))
        .and_then(|v| v.get("cells").and_then(Value::as_array).map(Vec::len))
        .unwrap_or(0)
}

fn twin_spike_total(branch_dir: &Path) -> Option<u64> {
    let doc = json_at(&branch_dir.join("TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS.json"))?;
    let cells = doc.get("cells")?.as_array()?;
    Some(
        cells
            .iter()
            .filter_map(|cell| cell.get("spike_count_total").and_then(Value::as_u64))
            .sum(),
    )
}

fn thermo_cells(branch_dir: &Path) -> usize {
    json_at(&branch_dir.join("thermodynamic_phase_2x2x2x2_condition_ledger.json"))
        .and_then(|v| v.get("cells").and_then(Value::as_array).map(Vec::len))
        .unwrap_or(0)
}

fn tensor_count(branch_dir: &Path) -> (usize, usize) {
    let Some(doc) = json_at(&branch_dir.join("dense_tensor_semantic_summaries.json")) else {
        return (0, 0);
    };
    let Some(rows) = doc.get("summaries").and_then(Value::as_array) else {
        return (0, 0);
    };
    let present = rows
        .iter()
        .filter(|row| row.get("summary_present").and_then(Value::as_bool) == Some(true))
        .count();
    (rows.len(), present)
}

fn number_path(value: &Value, path: &[&str]) -> Option<f64> {
    let mut cursor = value;
    for key in path {
        cursor = cursor.get(*key)?;
    }
    cursor.as_f64()
}

fn format_int_list(value: Option<&Value>, limit: usize) -> String {
    let Some(values) = value.and_then(Value::as_array) else {
        return "-".to_string();
    };
    let mut nums: Vec<String> = values
        .iter()
        .filter_map(Value::as_i64)
        .take(limit)
        .map(|v| v.to_string())
        .collect();
    if values.len() > limit {
        nums.push("...".to_string());
    }
    if nums.is_empty() {
        "-".to_string()
    } else {
        nums.join(",")
    }
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

fn text_cell(text: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.set_xalign(0.0);
    label.set_wrap(true);
    label
}

fn mono_cell(text: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.add_css_class("mono");
    label.set_xalign(0.0);
    label.set_wrap(false);
    label
}

fn status_pill(text: &str, pass: bool) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.add_css_class("status-pill");
    label.add_css_class(if pass { "pass" } else { "warn" });
    label.set_halign(gtk::Align::Start);
    label
}

fn fmt_f64(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "-".to_string())
}

fn fmt_u64(value: Option<u64>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn empty_state(title: &str, description: &str) -> gtk::Widget {
    adw::StatusPage::builder()
        .title(title)
        .description(description)
        .icon_name("dialog-information-symbolic")
        .build()
        .upcast()
}

fn science_tsv(rows: &[ScienceBranch]) -> String {
    let mut out = String::from(
        "branch\tdetection_mode\tligand_mode\tgdp_present\tbinding_sites\tpartial_sites\ttop_site\ttop_residues\ttop_family_rank\ttop_morphology_rank\ttop_physiology_rank\ttop_kcc_score\ttwin_cells\ttwin_pairwise_residuals\ttwin_factorial_status\ttwin_spike_support\tthermo_cells\tbridge_groups\ttensor_summaries\ttensor_summary_present\ttokenizer_records\ttokenizer_matched_fields\ttokenizer_gate\n",
    );
    for row in rows {
        let cells = [
            row.branch.clone(),
            row.detection_mode.clone(),
            row.ligand_mode.clone(),
            row.gdp_present.clone(),
            row.binding_sites.to_string(),
            row.partial_sites.to_string(),
            row.top_site.clone(),
            row.top_residues.clone(),
            fmt_f64(row.top_family_rank),
            fmt_f64(row.top_morphology_rank),
            fmt_f64(row.top_physiology_rank),
            fmt_f64(row.top_kcc_score),
            row.twin_cells.to_string(),
            fmt_u64(row.twin_pairwise_residuals),
            row.twin_factorial_status.clone(),
            fmt_u64(row.twin_spikes),
            row.thermo_cells.to_string(),
            fmt_u64(row.bridge_groups),
            row.tensor_summaries.to_string(),
            row.tensor_summary_present.to_string(),
            fmt_u64(row.tokenizer_records),
            fmt_u64(row.tokenizer_matched_fields),
            row.tokenizer_gate.clone(),
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

fn export_text(filename: &str, text: &str) -> std::io::Result<std::path::PathBuf> {
    let base = std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("PrismOS-Latest-Results")
        .join("exports");
    std::fs::create_dir_all(&base)?;
    let path = base.join(filename);
    std::fs::write(&path, text)?;
    Ok(path)
}

fn tsv_cell(value: &str) -> String {
    value
        .replace('\t', " ")
        .replace('\n', " ")
        .replace('\r', " ")
}
