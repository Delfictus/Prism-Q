//! Pocket Evolution — the per-POCKET occupancy + hydration evolution surface.
//!
//! This is the operator-facing view of `pocket_occupancy_evolution` (the DSTW tool
//! that assigns every REAL decoded spike to one of the materialized binding sites,
//! bins it on the real spike-timestep axis, and segments each pocket's occupancy
//! series with an offline BOCPD re-pass). It renders, per pocket:
//!   - a consensus occupancy + hydration table across the replicas, and
//!   - a per-pocket window timeline showing how occupancy / hydration evolve.
//!
//! Honesty is surfaced, not hidden: occupancy is a DERIVED spike-density proxy,
//! hydration (water_density) is REAL, per-pocket windows come from an offline BOCPD
//! substitute (the engine's own changepoints are on a different clock — shown as
//! metadata only), and geometric open/closed states are OMITTED (no coordinates).
//!
//! The full decode is heavy (~90s over ~37M spikes), so the tool is cache-first:
//! the panel loads instantly from cache and offers an explicit Recompute.

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;
use serde_json::Value;

use crate::adapters::PrismCommand;
use crate::app::AppContext;
use crate::jobs::JobEvent;

const TARGET: &str = "kras_6oim";

/// Build the Pocket Evolution panel.
pub fn build(ctx: &AppContext) -> gtk::Widget {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 12);
    root.add_css_class("prismos-root");
    root.set_margin_top(16);
    root.set_margin_bottom(16);
    root.set_margin_start(16);
    root.set_margin_end(16);

    let title = gtk::Label::new(Some("Pocket Evolution — per-pocket occupancy & hydration"));
    title.add_css_class("prismos-title");
    title.set_halign(gtk::Align::Start);
    root.append(&title);

    let subtitle = gtk::Label::new(Some(
        "Every detected pocket, resolved over the run's real spike-timestep axis: how its \
         occupancy (spike density) and hydration (water_density) evolve, segmented into states \
         by an offline BOCPD re-pass. Consensus across the 3 replicas; advisory only.",
    ));
    subtitle.add_css_class("subtitle");
    subtitle.set_halign(gtk::Align::Start);
    subtitle.set_wrap(true);
    root.append(&subtitle);

    // Static honesty/provenance legend — the labels the tool itself carries.
    let legend = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    legend.set_halign(gtk::Align::Start);
    legend.set_margin_top(2);
    for (cls, text) in [
        ("prov-real", "Hydration = REAL"),
        ("prov-derived", "Occupancy = DERIVED proxy"),
        ("prov-derived", "Windows = offline BOCPD"),
        ("prov-proxy", "hydrated/active = PROXY"),
        ("prov-omitted", "open/closed = OMITTED"),
    ] {
        let chip = gtk::Label::new(Some(text));
        chip.add_css_class("prov-chip");
        chip.add_css_class(cls);
        legend.append(&chip);
    }
    root.append(&legend);

    // Toolbar: status label + Recompute.
    let bar = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    bar.set_margin_top(4);
    let status = gtk::Label::new(Some("Loading pocket evolution\u{2026}"));
    status.add_css_class("subtitle");
    status.set_halign(gtk::Align::Start);
    status.set_hexpand(true);
    status.set_wrap(true);
    let recompute = gtk::Button::with_label("Recompute (~90s)");
    recompute.set_tooltip_text(Some(
        "Re-decode the full spike stream and rebuild the graph (heavy, ~90s).",
    ));
    bar.append(&status);
    bar.append(&recompute);
    root.append(&bar);

    let results = gtk::Box::new(gtk::Orientation::Vertical, 12);
    let scroller = gtk::ScrolledWindow::builder()
        .hexpand(true)
        .vexpand(true)
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&results)
        .build();
    root.append(&scroller);

    // Initial cache-first load.
    load(ctx, &results, &status, &recompute, false);

    {
        let ctx = ctx.clone();
        let results = results.clone();
        let status = status.clone();
        let recompute_c = recompute.clone();
        recompute.connect_clicked(move |_| load(&ctx, &results, &status, &recompute_c, true));
    }

    root.upcast()
}

fn load(
    ctx: &AppContext,
    results: &gtk::Box,
    status: &gtk::Label,
    button: &gtk::Button,
    refresh: bool,
) {
    button.set_sensitive(false);
    clear(results);
    status.set_text(if refresh {
        "Recomputing \u{2014} decoding the full spike stream (~90s)\u{2026}"
    } else {
        "Loading pocket evolution\u{2026}"
    });

    let cmd = PrismCommand::pocket_evolution_graph(TARGET, refresh);
    let acc = Rc::new(RefCell::new(String::new()));
    let results_c = results.clone();
    let status_c = status.clone();
    let button_c = button.clone();
    let ctx_c = ctx.clone();
    ctx.jobs.spawn(cmd, &ctx.resolver, move |ev| match ev {
        JobEvent::Stdout(line) => acc.borrow_mut().push_str(&line),
        JobEvent::Stderr(_) => {}
        JobEvent::Done(outcome) => {
            let raw = acc.borrow().clone();
            let rendered_ok = match serde_json::from_str::<Value>(raw.trim()) {
                Ok(json) => render(&results_c, &status_c, &json),
                Err(_) => {
                    status_c.set_text(if outcome.success {
                        "prism-pocket-evo returned no parseable JSON."
                    } else {
                        "prism-pocket-evo failed \u{2014} is the spike-decode shim built?"
                    });
                    false
                }
            };
            button_c.set_sensitive(true);
            if rendered_ok {
                ctx_c.toast("Pocket evolution ready");
            } else {
                ctx_c.toast("Pocket evolution failed \u{2014} is the spike-decode shim built?");
            }
        }
    });
}

/// Render the pocket-evolution graph. Returns `true` only when a real graph was
/// rendered; `false` when the tool reported a status/error field (early return),
/// so the caller can surface an honest failure toast instead of a false success.
fn render(results: &gtk::Box, status: &gtk::Label, json: &Value) -> bool {
    clear(results);

    if let Some(st) = json.get("status").and_then(Value::as_str) {
        let detail = json
            .get("detail")
            .and_then(Value::as_str)
            .unwrap_or("(no detail)");
        status.set_text(&format!("Tool status: {st} \u{2014} {detail}"));
        return false;
    }

    let label = json.get("label").and_then(Value::as_str).unwrap_or(TARGET);
    let run_id = json
        .get("sites_run_id")
        .and_then(Value::as_str)
        .unwrap_or("(unknown)");
    let n_sites = json
        .get("n_sites_defined")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let secs = json.get("_compute_seconds").and_then(Value::as_f64);
    let cached = secs.is_none();
    status.set_text(&format!(
        "{label}  \u{2022}  run {run_id}  \u{2022}  {n_sites} pockets  \u{2022}  {}",
        if cached {
            "loaded from cache".to_string()
        } else {
            format!("computed in {:.0}s", secs.unwrap_or(0.0))
        }
    ));

    // --- time-axis reconciliation banner (from rep_00) -----------------------
    if let Some(rep0) = json
        .get("replicas")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
    {
        results.append(&reconciliation_card(rep0));
    }

    // --- consensus per-site occupancy + hydration ----------------------------
    if let Some(cons) = json.get("consensus") {
        results.append(&consensus_card(cons));
    }

    // --- per-pocket window timelines (from rep_00) ---------------------------
    if let Some(rep0) = json
        .get("replicas")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
    {
        results.append(&timelines_card(rep0));
    }
    true
}

fn reconciliation_card(rep0: &Value) -> gtk::Widget {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 4);
    card.add_css_class("card");

    let head = section_label("How the states were anchored");
    card.append(&head);

    let recon = rep0.get("time_axis_reconciliation");
    let rstatus = recon
        .and_then(|r| r.get("reconciliation_status"))
        .and_then(Value::as_str)
        .unwrap_or("(unknown)");
    let ident = recon
        .and_then(|r| r.get("identity_map_check"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let src = recon
        .and_then(|r| r.get("primary_window_source"))
        .and_then(Value::as_str)
        .unwrap_or("");
    card.append(&text_row("Spike vs engine clock", rstatus));
    if !ident.is_empty() {
        card.append(&text_row("Identity check", ident));
    }
    card.append(&text_row("Per-pocket window edges", src));

    // Global engine changepoints — metadata only.
    if let Some(gc) = rep0.get("global_engine_changepoints") {
        let steps = gc
            .get("ste_changepoint_steps")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_i64)
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        card.append(&text_row(
            "Engine BOCPD (metadata only)",
            &format!(
                "steps [{steps}] \u{2014} GLOBAL, different clock; NOT used as per-pocket edges"
            ),
        ));
    }
    card.upcast()
}

fn consensus_card(cons: &Value) -> gtk::Widget {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 6);
    card.add_css_class("card");

    let n_reps = cons.get("n_replicas").and_then(Value::as_u64).unwrap_or(0);
    card.append(&section_label(&format!(
        "Consensus occupancy + hydration across {n_reps} replicas"
    )));

    let per_site = cons.get("per_site").and_then(Value::as_array);
    let max_occ = per_site
        .and_then(|a| a.first())
        .and_then(|s| s.get("mean_total_occupancy"))
        .and_then(Value::as_f64)
        .unwrap_or(1.0)
        .max(1.0);

    if let Some(sites) = per_site {
        for s in sites {
            card.append(&consensus_row(s, max_occ));
        }
    }
    card.upcast()
}

fn consensus_row(s: &Value, max_occ: f64) -> gtk::Widget {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);

    let site = s.get("site_id").and_then(Value::as_str).unwrap_or("site");
    let name = gtk::Label::new(Some(site));
    name.add_css_class("mono");
    name.set_width_chars(10);
    name.set_xalign(0.0);
    row.append(&name);

    let occ = s
        .get("mean_total_occupancy")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let bar = gtk::LevelBar::new();
    bar.set_min_value(0.0);
    bar.set_max_value(max_occ);
    bar.set_value(occ.clamp(0.0, max_occ));
    bar.set_hexpand(true);
    bar.set_valign(gtk::Align::Center);
    row.append(&bar);

    let sd = s
        .get("sd_total_occupancy")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let occ_lbl = gtk::Label::new(Some(&format!(
        "{} \u{00b1}{}",
        fmt_count(occ),
        fmt_count(sd)
    )));
    occ_lbl.add_css_class("mono");
    occ_lbl.set_width_chars(14);
    occ_lbl.set_xalign(1.0);
    row.append(&occ_lbl);

    let wd = s
        .get("mean_hydration_water_density")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let hyd = gtk::Label::new(Some(&format!("hyd {wd:.4}")));
    hyd.add_css_class("mono");
    hyd.set_width_chars(11);
    hyd.set_xalign(1.0);
    row.append(&hyd);

    let repro = s
        .get("consensus_reproducible")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let badge = gtk::Label::new(Some(if repro { "3/3" } else { "partial" }));
    badge.add_css_class("prov-chip");
    badge.add_css_class(if repro { "prov-real" } else { "prov-proxy" });
    badge.set_width_chars(8);
    row.append(&badge);

    row.upcast()
}

fn timelines_card(rep0: &Value) -> gtk::Widget {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 8);
    card.add_css_class("card");
    card.append(&section_label(
        "Per-pocket evolution (rep_00 \u{2014} windows over the real spike-time axis)",
    ));

    let nodes = match rep0.get("nodes").and_then(Value::as_array) {
        Some(n) => n,
        None => return card.upcast(),
    };

    // Group nodes by site_id, preserving encounter order (windows are appended in order).
    let mut order: Vec<String> = Vec::new();
    let mut groups: std::collections::HashMap<String, Vec<&Value>> =
        std::collections::HashMap::new();
    for node in nodes {
        let sid = node
            .get("site_id")
            .and_then(Value::as_str)
            .unwrap_or("site")
            .to_string();
        if !groups.contains_key(&sid) {
            order.push(sid.clone());
        }
        groups.entry(sid).or_default().push(node);
    }

    for sid in &order {
        let windows = &groups[sid];
        let site_row = gtk::Box::new(gtk::Orientation::Vertical, 3);
        site_row.set_margin_top(4);

        let head = gtk::Label::new(Some(&format!("{sid}  ({} windows)", windows.len())));
        head.add_css_class("pocket-site-label");
        head.set_halign(gtk::Align::Start);
        site_row.append(&head);

        let max_n = windows
            .iter()
            .filter_map(|w| w.get("n_spikes").and_then(Value::as_f64))
            .fold(1.0_f64, f64::max);

        let strip = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        for w in windows {
            strip.append(&window_cell(w, max_n));
        }
        let hscroll = gtk::ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Automatic)
            .vscrollbar_policy(gtk::PolicyType::Never)
            .child(&strip)
            .build();
        hscroll.set_hexpand(true);
        site_row.append(&hscroll);

        card.append(&site_row);
    }
    card.upcast()
}

fn window_cell(w: &Value, max_n: f64) -> gtk::Widget {
    let cell = gtk::Box::new(gtk::Orientation::Vertical, 2);
    cell.add_css_class("pocket-cell");

    let labels = w.get("proxy_state_labels");
    let hydrated = labels
        .and_then(|l| l.get("hydrated"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let active = labels
        .and_then(|l| l.get("active"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if hydrated {
        cell.add_css_class("pocket-cell-hydrated");
    }
    if active {
        cell.add_css_class("pocket-cell-active");
    }

    let f0 = w.get("frame_start").and_then(Value::as_i64).unwrap_or(0);
    let f1 = w.get("frame_end").and_then(Value::as_i64).unwrap_or(0);
    let head = gtk::Label::new(Some(&format!("[{f0},{f1})")));
    head.add_css_class("mono");
    head.set_xalign(0.0);
    cell.append(&head);

    let n = w.get("n_spikes").and_then(Value::as_f64).unwrap_or(0.0);
    let bar = gtk::LevelBar::new();
    bar.set_min_value(0.0);
    bar.set_max_value(max_n.max(1.0));
    bar.set_value(n.clamp(0.0, max_n));
    bar.set_size_request(120, 6);
    cell.append(&bar);

    let occ = gtk::Label::new(Some(&format!("occ {}", fmt_count(n))));
    occ.add_css_class("mono");
    occ.set_xalign(0.0);
    cell.append(&occ);

    let wd = w
        .get("mean_water_density")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let hyd = gtk::Label::new(Some(&format!("wd {wd:.4}")));
    hyd.add_css_class("mono");
    hyd.set_xalign(0.0);
    cell.append(&hyd);

    let tags = gtk::Label::new(Some(&format!(
        "{} {}",
        if hydrated {
            "\u{25cf}HYDRATED"
        } else {
            "\u{25cb}dry"
        },
        if active {
            "\u{25cf}ACTIVE"
        } else {
            "\u{25cb}quiet"
        },
    )));
    tags.add_css_class("subtitle");
    tags.set_xalign(0.0);
    cell.append(&tags);

    cell.upcast()
}

// --- helpers ----------------------------------------------------------------

fn section_label(text: &str) -> gtk::Label {
    let l = gtk::Label::new(Some(text));
    l.add_css_class("copilot-tools");
    l.set_halign(gtk::Align::Start);
    l
}

fn text_row(label: &str, value: &str) -> gtk::Box {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let l = gtk::Label::new(Some(label));
    l.set_width_chars(26);
    l.set_xalign(0.0);
    let v = gtk::Label::new(Some(value));
    v.set_xalign(0.0);
    v.set_hexpand(true);
    v.set_wrap(true);
    v.set_selectable(true);
    row.append(&l);
    row.append(&v);
    row
}

/// Compact human count: 864109 -> "864.1k", 1_540_804 -> "1.5M".
fn fmt_count(x: f64) -> String {
    let a = x.abs();
    if a >= 1.0e6 {
        format!("{:.1}M", x / 1.0e6)
    } else if a >= 1.0e3 {
        format!("{:.1}k", x / 1.0e3)
    } else {
        format!("{x:.0}")
    }
}

fn clear(container: &gtk::Box) {
    while let Some(child) = container.first_child() {
        container.remove(&child);
    }
}
