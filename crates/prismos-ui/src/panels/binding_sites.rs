//! Path-B / V2 binding-site panel.
//!
//! This is the direct operator surface for `binding_sites.materialized.json`.
//! It loads the four native 4LPK modality branches from the active run root and
//! renders the actual materialized site records in-app, without making the user
//! hunt through the Evidence browser.

use std::path::{Path, PathBuf};

use adw::prelude::*;
use serde_json::Value;

use crate::app::AppContext;
use crate::model::run_layout;

#[derive(Debug, Clone)]
struct BranchSummary {
    branch: String,
    source: PathBuf,
    status: String,
    site_count: usize,
    partial_count: usize,
    raw_peak_count: Option<u64>,
    consolidated_count: Option<u64>,
}

#[derive(Debug, Clone)]
struct SiteRow {
    branch: String,
    site_id: String,
    rank: i64,
    promoted: bool,
    centroid_xyz: String,
    lining_residues: String,
    driver_residues: String,
    representative_residues: String,
    final_prism_score: Option<f64>,
    family_rank: Option<f64>,
    morphology_rank: Option<f64>,
    physiology_rank: Option<f64>,
    kcc_score: Option<f64>,
    temporal_persistence: Option<f64>,
    dynamic_support: Option<f64>,
    stream_count: Option<u64>,
    spike_count: Option<u64>,
    materialization_level: String,
    limitations: String,
}

/// Build the binding-site panel and refresh it whenever the page maps, so run
/// root changes made elsewhere in the app are reflected immediately.
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
            "Binding sites unavailable",
            "No run root is selected. Open Run Root and select a PrismOS native run.",
        );
    };

    let loaded = load_binding_sites(&run_dir);
    let total_sites = loaded.sites.len();
    let native_publications: Vec<PathBuf> = run_layout::discover_binding_site_branches(&run_dir).iter()
        .flat_map(|branch| run_layout::native_binding_site_publications(&branch.path)).collect();
    if loaded.summaries.is_empty() && native_publications.is_empty() {
        return empty_state(
            "Binding sites unavailable",
            "No materialized binding-site document or canonical native binding-site publication was found in this run root.",
        );
    }

    let root = gtk::Box::new(gtk::Orientation::Vertical, 14);
    root.set_margin_top(16);
    root.set_margin_bottom(16);
    root.set_margin_start(16);
    root.set_margin_end(16);

    for publication in native_publications { root.append(&native_publication_card(ctx, &publication)); }
    if !loaded.summaries.is_empty() {
        root.append(&header_card(&run_dir, &loaded.summaries, total_sites));
        root.append(&action_bar(ctx, &loaded.sites));
        root.append(&branch_summary_grid(&loaded.summaries));
        root.append(&site_table(&loaded.sites));
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

/// Distinct native source-format view: never coerced into legacy materialized scores.
fn native_publication_card(ctx: &AppContext, path: &Path) -> gtk::Widget {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 10);
    card.add_css_class("card");
    let loaded = match run_layout::load_native_binding_publication(path) {
        Ok(value) => value,
        Err(error) => {
            let text = gtk::Label::new(Some(&format!("Native publication could not be verified: {error}\n{}\nOriginal files remain unchanged.", path.display())));
            text.set_wrap(true); text.set_selectable(true); text.set_xalign(0.0); card.append(&text);
            return card.upcast();
        }
    };
    let title = gtk::Label::new(Some(&format!("{} — {} native binding sites", loaded.structure, loaded.sites.len())));
    title.add_css_class("prismos-title"); title.set_xalign(0.0); card.append(&title);
    let provenance = gtk::Label::new(Some(&format!("Native identity domain: {}\nSource: {}\nSHA256: {}\nPublication: {}\nSource/count/IDs/ranks/geometry and {} visual-file hashes verified. This checks publication consistency; scientific acceptance remains in the production receipts.",
        loaded.identity_domain, loaded.source.display(), loaded.source_sha256, loaded.publication.display(), loaded.visual_files.len())));
    provenance.set_wrap(true); provenance.set_selectable(true); provenance.set_xalign(0.0); card.append(&provenance);
    let scope = gtk::Label::new(Some("Publication order is the final native sites array order. Native rank, cryptic_rank, quality_score and druggability retain separate meanings; missing values stay unavailable. Native lining-residue objects and all additional fields are retained in each complete record."));
    scope.set_wrap(true); scope.set_xalign(0.0); card.append(&scope);
    let copy = gtk::Button::with_label("Copy complete native inventory as TSV");
    let mut tsv = String::from("publication_order\tnative_id\trank\tcryptic_rank\tcentroid\tquality_score\tdruggability\tclassification\ttherm_class\tlining_residues\tcomplete_native_record\tsource_sha256\n");
    for (index, site) in loaded.sites.iter().enumerate() {
        let mut cells = vec![(index + 1).to_string()];
        for field in ["id", "rank", "cryptic_rank", "centroid", "quality_score", "druggability", "classification", "therm_class", "lining_residues"] {
            cells.push(site[field].to_string());
        }
        cells.push(site.to_string()); cells.push(loaded.source_sha256.clone());
        tsv.push_str(&cells.iter().map(|s| tsv_cell(s)).collect::<Vec<_>>().join("\t")); tsv.push('\n');
        let title = format!("Order {} • native ID {} • rank {} • cryptic_rank {} • quality_score {} • druggability {}",
            index + 1, site["id"], site["rank"], site["cryptic_rank"], site["quality_score"], site["druggability"]);
        let details = gtk::Expander::builder().label(&title).build();
        let text = gtk::TextView::new(); text.set_editable(false); text.set_cursor_visible(false); text.set_monospace(true);
        text.buffer().set_text(&serde_json::to_string_pretty(site).unwrap_or_else(|_| site.to_string()));
        details.set_child(Some(&text)); card.append(&details);
    }
    let ctx = ctx.clone(); copy.connect_clicked(move |_| { copy_text(&tsv); ctx.toast("Complete native site inventory copied; native fields and order preserved"); });
    card.append(&copy);
    card.upcast()
}

struct LoadedSites {
    summaries: Vec<BranchSummary>,
    sites: Vec<SiteRow>,
}

fn load_binding_sites(run_dir: &Path) -> LoadedSites {
    let mut summaries = Vec::new();
    let mut sites = Vec::new();

    // Discover branch directories from the on-disk evidence rather than a
    // hardcoded name list: any directory whose subtree carries a
    // `pathb_runtime_stage/binding_sites.materialized.json`.
    for branch_info in run_layout::discover_binding_site_branches(run_dir) {
        let source = branch_info.binding_sites_materialized();
        let branch = branch_info.name.as_str();
        let Ok(text) = std::fs::read_to_string(&source) else {
            continue;
        };
        let Ok(doc) = serde_json::from_str::<Value>(&text) else {
            continue;
        };

        let branch_sites = doc
            .get("binding_sites")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let partial_count = doc
            .get("spatiotemporal_partial_candidates")
            .and_then(Value::as_array)
            .map(|rows| rows.len())
            .unwrap_or(0);
        let status = doc
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        summaries.push(BranchSummary {
            branch: branch.to_string(),
            source: source.clone(),
            status,
            site_count: branch_sites.len(),
            partial_count,
            raw_peak_count: doc.get("n_raw_peaks").and_then(Value::as_u64),
            consolidated_count: doc.get("n_consolidated_regions").and_then(Value::as_u64),
        });

        for site in branch_sites {
            sites.push(parse_site_row(branch, &source, &site));
        }
    }

    sites.sort_by(|a, b| {
        a.branch
            .cmp(&b.branch)
            .then_with(|| a.rank.cmp(&b.rank))
            .then_with(|| {
                b.family_rank
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(&a.family_rank.unwrap_or(f64::NEG_INFINITY))
            })
    });

    LoadedSites { summaries, sites }
}

fn parse_site_row(branch: &str, _source: &Path, site: &Value) -> SiteRow {
    let site_id = str_field(site, "site_id").unwrap_or_else(|| "site".to_string());
    let rank = int_field(site, "rank")
        .or_else(|| int_field(site, "new_prism_rank"))
        .unwrap_or(0);

    let final_prism_score = number_path(site, &["lexicographic_rank_key", "final_prism_score"])
        .or_else(|| number_path(site, &["score_components", "final_prism_score"]));
    let family_rank = number_path(
        site,
        &["lexicographic_rank_key", "site_family_preservation_rank"],
    )
    .or_else(|| {
        number_path(
            site,
            &["site_family_preservation", "site_family_preservation_rank"],
        )
    });
    let morphology_rank = number_path(site, &["lexicographic_rank_key", "pocket_morphology_rank"])
        .or_else(|| number_path(site, &["pocket_morphology_surface", "morphology_rank"]));
    let physiology_rank = number_path(
        site,
        &["lexicographic_rank_key", "thermodynamic_physiology_rank"],
    )
    .or_else(|| {
        number_path(
            site,
            &["binding_site_physiology_surface", "physiology_rank"],
        )
    });
    let kcc_score = number_path(site, &["kcc_driver", "kcc_driver_score"]);
    let temporal_persistence = number_path(
        site,
        &["lexicographic_rank_key", "temporal_persistence_factor"],
    )
    .or_else(|| number_path(site, &["refined_temporal_support", "persistence_fraction"]));
    let dynamic_support = number_path(
        site,
        &["lexicographic_rank_key", "scale_normalized_dynamic_support"],
    )
    .or_else(|| {
        number_path(
            site,
            &[
                "binding_site_physiology_surface",
                "scale_normalized_dynamic_support",
            ],
        )
    });

    SiteRow {
        branch: branch.to_string(),
        site_id,
        rank,
        promoted: site
            .get("promoted")
            .and_then(Value::as_bool)
            .or_else(|| {
                site.pointer("/lexicographic_rank_key/promoted")
                    .and_then(Value::as_bool)
            })
            .unwrap_or(false),
        centroid_xyz: format_xyz(site.get("centroid_xyz")),
        lining_residues: format_int_list(site.get("lining_residues"), 18),
        driver_residues: format_int_list(site.get("driver_residues"), 12),
        representative_residues: format_int_list(site.get("representative_residues"), 8),
        final_prism_score,
        family_rank,
        morphology_rank,
        physiology_rank,
        kcc_score,
        temporal_persistence,
        dynamic_support,
        stream_count: site
            .pointer("/stream_support/n_streams_supporting")
            .and_then(Value::as_u64),
        spike_count: site.get("n_spikes").and_then(Value::as_u64),
        materialization_level: str_field(site, "materialization_level")
            .unwrap_or_else(|| "unknown".to_string()),
        limitations: format_string_list(site.get("limitations"), 4),
    }
}

fn header_card(run_dir: &Path, summaries: &[BranchSummary], total_sites: usize) -> gtk::Widget {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 10);
    card.add_css_class("card");

    let title = gtk::Label::new(Some("Binding Sites"));
    title.add_css_class("prismos-title");
    title.set_xalign(0.0);
    card.append(&title);

    let stats = gtk::Box::new(gtk::Orientation::Horizontal, 28);
    stats.append(&stat(&total_sites.to_string(), "Materialized sites"));
    stats.append(&stat(&summaries.len().to_string(), "Runtime branches"));
    let partials: usize = summaries.iter().map(|s| s.partial_count).sum();
    stats.append(&stat(&partials.to_string(), "Partial candidates"));
    card.append(&stats);

    let source = gtk::Label::new(Some(&format!("run root: {}", run_dir.display())));
    source.add_css_class("dim");
    source.add_css_class("mono");
    source.set_xalign(0.0);
    source.set_wrap(true);
    card.append(&source);

    card.upcast()
}

fn action_bar(ctx: &AppContext, sites: &[SiteRow]) -> gtk::Widget {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    row.set_halign(gtk::Align::Start);

    let copy_btn = gtk::Button::with_label("Copy TSV");
    copy_btn.add_css_class("accent-button");
    let export_btn = gtk::Button::with_label("Export TSV");

    let tsv = binding_sites_tsv(sites);
    {
        let ctx = ctx.clone();
        let tsv = tsv.clone();
        copy_btn.connect_clicked(move |_| {
            copy_text(&tsv);
            ctx.toast("Binding sites copied as TSV");
        });
    }
    {
        let ctx = ctx.clone();
        let tsv = tsv.clone();
        export_btn.connect_clicked(move |_| match export_text("binding_sites.tsv", &tsv) {
            Ok(path) => ctx.toast(&format!("Exported binding sites: {}", path.display())),
            Err(err) => ctx.toast(&format!("Export failed: {err}")),
        });
    }

    row.append(&copy_btn);
    row.append(&export_btn);
    row.upcast()
}

fn branch_summary_grid(summaries: &[BranchSummary]) -> gtk::Widget {
    let grid = gtk::Grid::new();
    grid.set_column_spacing(18);
    grid.set_row_spacing(8);
    grid.set_margin_top(10);
    grid.set_margin_bottom(10);
    grid.set_margin_start(10);
    grid.set_margin_end(10);

    for (col, title) in [
        "Branch",
        "Status",
        "Sites",
        "Partials",
        "Raw peaks",
        "Regions",
        "Source",
    ]
    .iter()
    .enumerate()
    {
        grid.attach(&header_cell(title), col as i32, 0, 1, 1);
    }

    for (idx, summary) in summaries.iter().enumerate() {
        let r = (idx + 1) as i32;
        grid.attach(&mono_cell(&summary.branch), 0, r, 1, 1);
        grid.attach(&text_cell(&summary.status), 1, r, 1, 1);
        grid.attach(&mono_cell(&summary.site_count.to_string()), 2, r, 1, 1);
        grid.attach(&mono_cell(&summary.partial_count.to_string()), 3, r, 1, 1);
        grid.attach(&mono_cell(&opt_u64(summary.raw_peak_count)), 4, r, 1, 1);
        grid.attach(&mono_cell(&opt_u64(summary.consolidated_count)), 5, r, 1, 1);
        grid.attach(&source_cell(&summary.source), 6, r, 1, 1);
    }

    let scroller = gtk::ScrolledWindow::builder()
        .min_content_height(145)
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Never)
        .child(&grid)
        .build();
    scroller.add_css_class("card");
    scroller.upcast()
}

fn site_table(sites: &[SiteRow]) -> gtk::Widget {
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
        "Promoted",
        "Centroid",
        "Lining residues",
        "Driver residues",
        "Representative",
        "Final",
        "Family rank",
        "Morphology",
        "Physiology",
        "KCC",
        "Temporal",
        "Dynamic",
        "Streams",
        "Spikes",
        "Level",
        "Limitations",
    ];
    for (col, title) in headers.iter().enumerate() {
        grid.attach(&header_cell(title), col as i32, 0, 1, 1);
    }

    for (idx, site) in sites.iter().enumerate() {
        let r = (idx + 1) as i32;
        grid.attach(&mono_cell(&site.branch), 0, r, 1, 1);
        grid.attach(&mono_cell(&site.site_id), 1, r, 1, 1);
        grid.attach(&mono_cell(&site.rank.to_string()), 2, r, 1, 1);
        grid.attach(
            &status_pill(if site.promoted { "YES" } else { "NO" }, site.promoted),
            3,
            r,
            1,
            1,
        );
        grid.attach(&mono_cell(&site.centroid_xyz), 4, r, 1, 1);
        grid.attach(&mono_cell(&site.lining_residues), 5, r, 1, 1);
        grid.attach(&mono_cell(&site.driver_residues), 6, r, 1, 1);
        grid.attach(&mono_cell(&site.representative_residues), 7, r, 1, 1);
        grid.attach(&mono_cell(&opt_f64(site.final_prism_score)), 8, r, 1, 1);
        grid.attach(&mono_cell(&opt_f64(site.family_rank)), 9, r, 1, 1);
        grid.attach(&mono_cell(&opt_f64(site.morphology_rank)), 10, r, 1, 1);
        grid.attach(&mono_cell(&opt_f64(site.physiology_rank)), 11, r, 1, 1);
        grid.attach(&mono_cell(&opt_f64(site.kcc_score)), 12, r, 1, 1);
        grid.attach(&mono_cell(&opt_f64(site.temporal_persistence)), 13, r, 1, 1);
        grid.attach(&mono_cell(&opt_f64(site.dynamic_support)), 14, r, 1, 1);
        grid.attach(&mono_cell(&opt_u64(site.stream_count)), 15, r, 1, 1);
        grid.attach(&mono_cell(&opt_u64(site.spike_count)), 16, r, 1, 1);
        grid.attach(&text_cell(&site.materialization_level), 17, r, 1, 1);
        grid.attach(&text_cell(&site.limitations), 18, r, 1, 1);
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

fn empty_state(title: &str, description: &str) -> gtk::Widget {
    adw::StatusPage::builder()
        .title(title)
        .description(description)
        .icon_name("dialog-information-symbolic")
        .build()
        .upcast()
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

fn source_cell(path: &Path) -> gtk::Label {
    let label = mono_cell(&path.display().to_string());
    label.add_css_class("dim");
    label.set_selectable(true);
    label.set_ellipsize(gtk::pango::EllipsizeMode::Start);
    label
}

fn status_pill(text: &str, pass: bool) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.add_css_class("status-pill");
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

fn str_field(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

fn int_field(value: &Value, key: &str) -> Option<i64> {
    value.get(key).and_then(Value::as_i64)
}

fn number_path(value: &Value, path: &[&str]) -> Option<f64> {
    let mut cursor = value;
    for key in path {
        cursor = cursor.get(*key)?;
    }
    cursor.as_f64()
}

fn format_xyz(value: Option<&Value>) -> String {
    let Some(values) = value.and_then(Value::as_array) else {
        return "-".to_string();
    };
    let nums: Vec<String> = values
        .iter()
        .filter_map(Value::as_f64)
        .take(3)
        .map(|v| format!("{v:.2}"))
        .collect();
    if nums.len() == 3 {
        nums.join(", ")
    } else {
        "-".to_string()
    }
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

fn format_string_list(value: Option<&Value>, limit: usize) -> String {
    let Some(values) = value.and_then(Value::as_array) else {
        return String::new();
    };
    let mut strings: Vec<String> = values
        .iter()
        .filter_map(Value::as_str)
        .take(limit)
        .map(str::to_string)
        .collect();
    if values.len() > limit {
        strings.push("...".to_string());
    }
    strings.join(", ")
}

fn binding_sites_tsv(sites: &[SiteRow]) -> String {
    let mut out = String::from(
        "branch\tsite_id\trank\tpromoted\tcentroid_xyz\tlining_residues\tdriver_residues\trepresentative_residues\tfinal_prism_score\tfamily_rank\tmorphology_rank\tphysiology_rank\tkcc_score\ttemporal_persistence\tdynamic_support\tstream_count\tspike_count\tmaterialization_level\tlimitations\n",
    );
    for site in sites {
        let row = [
            site.branch.clone(),
            site.site_id.clone(),
            site.rank.to_string(),
            site.promoted.to_string(),
            site.centroid_xyz.clone(),
            site.lining_residues.clone(),
            site.driver_residues.clone(),
            site.representative_residues.clone(),
            opt_f64(site.final_prism_score),
            opt_f64(site.family_rank),
            opt_f64(site.morphology_rank),
            opt_f64(site.physiology_rank),
            opt_f64(site.kcc_score),
            opt_f64(site.temporal_persistence),
            opt_f64(site.dynamic_support),
            opt_u64(site.stream_count),
            opt_u64(site.spike_count),
            site.materialization_level.clone(),
            site.limitations.clone(),
        ];
        out.push_str(
            &row.iter()
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

fn tsv_cell(value: &str) -> String {
    value
        .replace('\t', " ")
        .replace('\n', " ")
        .replace('\r', " ")
}
