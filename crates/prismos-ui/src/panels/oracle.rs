//! PrismAI — the champion physics model surface.
//!
//! Enter a canonical SMILES (or upload a molecule file), and the local champion
//! (Qwen3-8B + PRISM fusion, :8711) returns its physics envelope, rendered as a
//! visual readout PLUS a plain-language interpretation of what confidence,
//! thermodynamic certainty, entropy, and kinetic timescale actually MEAN for
//! this molecule. Molecules accumulate as cards and can be stacked side-by-side
//! in a compare grid. Advisory / physics only — this is NOT a chat (that is
//! Prism CoPilot); no canonical mutation.

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;
use serde_json::Value;

use crate::adapters::PrismCommand;
use crate::app::AppContext;
use crate::jobs::JobEvent;

/// One queried molecule's stored physics summary (for the compare grid). Every
/// field is a REAL value returned by the physics server (or `None` when the
/// server did not return it) — no composite score, no fabricated default.
struct MolRecord {
    smiles: String,
    #[allow(dead_code)]
    provenance: Option<String>,
    confidence: f64,
    certainty: f64,
    entropy_nats: Option<f64>,
    time_ps: Option<f64>,
    context_len: Option<i64>,
    #[allow(dead_code)]
    json: Value,
}

impl MolRecord {
    fn from_json(smiles: &str, provenance: Option<String>, json: &Value) -> Self {
        Self {
            smiles: smiles.to_string(),
            provenance,
            confidence: f64_of(json, &["confidence"]).unwrap_or(0.0),
            certainty: f64_of(json, &["latent_scores", "certainty"]).unwrap_or(0.0),
            entropy_nats: f64_of(json, &["latent_scores", "entropy_nats"]),
            time_ps: f64_of(json, &["decoded", "time_ps_estimate"])
                .or_else(|| f64_of(json, &["decoded", "time_ps"])),
            context_len: json.get("context_len").and_then(|x| x.as_i64()),
            json: json.clone(),
        }
    }
}

/// Shared panel state handed to every closure.
struct Oracle {
    ctx: AppContext,
    results: gtk::Box,
    compare_container: gtk::Box,
    entry: gtk::Entry,
    predict_btn: gtk::Button,
    compare_toggle: gtk::ToggleButton,
    prov_label: gtk::Label,
    records: RefCell<Vec<MolRecord>>,
}

/// Build the PrismAI physics panel.
pub fn build(ctx: &AppContext) -> gtk::Widget {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 12);
    root.add_css_class("prismos-root");
    root.set_margin_top(16);
    root.set_margin_bottom(16);
    root.set_margin_start(16);
    root.set_margin_end(16);

    let title = gtk::Label::new(Some("PrismAI — Physics Engine"));
    title.add_css_class("prismos-title");
    title.set_halign(gtk::Align::Start);
    root.append(&title);

    let subtitle = gtk::Label::new(Some(
        "Enter a canonical SMILES or upload a molecule file. The local champion physics model \
         returns its envelope — next-state confidence, thermodynamic certainty, entropy, kinetic \
         timescale — rendered visually and in plain language. Stack molecules to compare them. \
         Advisory / physics only (not a chat — that is Prism CoPilot).",
    ));
    subtitle.add_css_class("subtitle");
    subtitle.set_halign(gtk::Align::Start);
    subtitle.set_wrap(true);
    root.append(&subtitle);

    let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let entry = gtk::Entry::new();
    entry.set_placeholder_text(Some(
        "canonical SMILES, e.g. Cc1ccc2[nH]cc(CC[C@@H]3CNCC3=O)c2c1",
    ));
    entry.set_hexpand(true);
    let upload = gtk::Button::with_label("Upload molecule\u{2026}");
    let button = gtk::Button::with_label("Predict Physics");
    button.add_css_class("accent-button");
    let clear = gtk::Button::with_label("Clear");
    row.append(&entry);
    row.append(&upload);
    row.append(&button);
    row.append(&clear);
    root.append(&row);

    // provenance chip (hidden until a SMILES is derived from a file)
    let prov_label = gtk::Label::new(None);
    prov_label.add_css_class("prov-chip");
    prov_label.add_css_class("prov-derived");
    prov_label.set_halign(gtk::Align::Start);
    prov_label.set_wrap(true);
    prov_label.set_xalign(0.0);
    prov_label.set_visible(false);
    root.append(&prov_label);

    // compare toggle row
    let compare_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let compare_toggle = gtk::ToggleButton::with_label("Compare (0)");
    compare_toggle.set_tooltip_text(Some(
        "Show a side-by-side grid of all queried molecules (needs \u{2265} 2).",
    ));
    compare_row.append(&compare_toggle);
    root.append(&compare_row);

    // scroller content: compare card container above the molecule cards
    let content = gtk::Box::new(gtk::Orientation::Vertical, 10);
    let compare_container = gtk::Box::new(gtk::Orientation::Vertical, 0);
    let results = gtk::Box::new(gtk::Orientation::Vertical, 10);
    let placeholder = gtk::Label::new(Some(
        "Enter a SMILES or upload a molecule, then Predict Physics.",
    ));
    placeholder.add_css_class("subtitle");
    placeholder.set_halign(gtk::Align::Start);
    results.append(&placeholder);
    content.append(&compare_container);
    content.append(&results);
    let scroller = gtk::ScrolledWindow::builder()
        .hexpand(true)
        .vexpand(true)
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&content)
        .build();
    root.append(&scroller);

    let ui = Rc::new(Oracle {
        ctx: ctx.clone(),
        results: results.clone(),
        compare_container: compare_container.clone(),
        entry: entry.clone(),
        predict_btn: button.clone(),
        compare_toggle: compare_toggle.clone(),
        prov_label: prov_label.clone(),
        records: RefCell::new(Vec::new()),
    });

    // predict button + entry activate
    {
        let ui = ui.clone();
        button.connect_clicked(move |_| run_query(&ui, None));
    }
    {
        let ui = ui.clone();
        entry.connect_activate(move |_| run_query(&ui, None));
    }
    // upload molecule file
    {
        let ui = ui.clone();
        upload.connect_clicked(move |_| open_molecule(&ui));
    }
    // clear: empties cards, records, compare display, prov chip
    {
        let ui = ui.clone();
        clear.connect_clicked(move |_| {
            while let Some(child) = ui.results.first_child() {
                ui.results.remove(&child);
            }
            ui.records.borrow_mut().clear();
            while let Some(child) = ui.compare_container.first_child() {
                ui.compare_container.remove(&child);
            }
            ui.compare_toggle.set_active(false);
            update_compare_label(&ui);
            ui.prov_label.set_visible(false);
            ui.prov_label.set_text("");
        });
    }
    // compare toggle
    {
        let ui = ui.clone();
        compare_toggle.connect_toggled(move |_| rebuild_compare(&ui));
    }

    root.upcast()
}

/// Open a molecule file and derive a SMILES. `.smi` is parsed in pure Rust;
/// `.sdf`/`.mol`/`.pdb` route through the `prism-mol2smi` chem-toolkit wrapper,
/// which degrades honestly when no toolkit is present.
fn open_molecule(ui: &Rc<Oracle>) {
    let dialog = gtk::FileDialog::new();
    dialog.set_title("Upload a molecule (.smi / .sdf / .mol / .pdb)");
    let filter = gtk::FileFilter::new();
    filter.set_name(Some("Molecule files (*.smi, *.sdf, *.mol, *.pdb)"));
    for pat in ["*.smi", "*.sdf", "*.mol", "*.pdb"] {
        filter.add_pattern(pat);
    }
    dialog.set_default_filter(Some(&filter));
    let ui = ui.clone();
    let window = ui.ctx.window.clone();
    dialog.open(Some(&window), gtk::gio::Cancellable::NONE, move |result| {
        let Ok(file) = result else { return };
        let Some(path) = file.path() else { return };
        let fname = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        match ext.as_str() {
            "smi" => match std::fs::read_to_string(&path) {
                Ok(text) => {
                    let tok = text
                        .lines()
                        .map(str::trim)
                        .find(|l| !l.is_empty() && !l.starts_with('#'))
                        .and_then(|l| l.split_whitespace().next());
                    match tok {
                        Some(s) if !s.is_empty() => {
                            ui.entry.set_text(s);
                            run_query(
                                &ui,
                                Some(format!("SMILES read directly from {fname} (.smi token 0)")),
                            );
                        }
                        _ => ui.results.prepend(&error_card(
                            &fname,
                            "empty / unparseable .smi (no SMILES token)",
                        )),
                    }
                }
                Err(e) => ui
                    .results
                    .prepend(&error_card(&fname, &format!("could not read file: {e}"))),
            },
            "sdf" | "mol" | "pdb" => derive_via_toolkit(&ui, &path, &fname),
            other => ui.results.prepend(&error_card(
                &fname,
                &format!("unsupported file type .{other} — use .smi / .sdf / .mol / .pdb"),
            )),
        }
    });
}

/// Stream `prism-mol2smi <file>` and, on success, fill the (editable) entry so
/// the operator can confirm the perceived SMILES before physics runs.
fn derive_via_toolkit(ui: &Rc<Oracle>, path: &std::path::Path, fname: &str) {
    let cmd = PrismCommand::mol_to_smiles(path);
    let acc = Rc::new(RefCell::new(String::new()));
    let ui = ui.clone();
    let fname = fname.to_string();
    let jobs = ui.ctx.jobs.clone();
    let resolver = ui.ctx.resolver.clone();
    jobs.spawn(cmd, &resolver, move |ev| match ev {
        JobEvent::Stdout(line) => acc.borrow_mut().push_str(&line),
        JobEvent::Stderr(_) => {}
        JobEvent::Done(_) => {
            let raw = acc.borrow().clone();
            match serde_json::from_str::<Value>(raw.trim()) {
                Ok(j) => {
                    if let Some(s) = j.get("smiles").and_then(|x| x.as_str()) {
                        let method = j
                            .get("method")
                            .and_then(|x| x.as_str())
                            .unwrap_or("chem toolkit");
                        ui.entry.set_text(s);
                        run_query(
                            &ui,
                            Some(format!("from {fname} via {method} — verify perception")),
                        );
                    } else if let Some(err) = j.get("error").and_then(|x| x.as_str()) {
                        let install = j.get("install").and_then(|x| x.as_str());
                        let msg = match install {
                            Some(i) if !i.is_empty() => format!("{err} — {i}"),
                            _ => err.to_string(),
                        };
                        ui.results.prepend(&error_card(&fname, &msg));
                    } else {
                        ui.results
                            .prepend(&error_card(&fname, "prism-mol2smi returned no smiles"));
                    }
                }
                Err(_) => ui.results.prepend(&error_card(
                    &fname,
                    "prism-mol2smi returned unparseable output (chem toolkit missing?)",
                )),
            }
        }
    });
}

fn run_query(ui: &Rc<Oracle>, provenance: Option<String>) {
    let smiles = ui.entry.text().to_string().trim().to_string();
    if smiles.is_empty() {
        return;
    }
    // provenance chip
    match &provenance {
        Some(p) => {
            ui.prov_label.set_text(p);
            ui.prov_label.set_visible(true);
        }
        None => {
            ui.prov_label.set_text("");
            ui.prov_label.set_visible(false);
        }
    }
    ui.predict_btn.set_sensitive(false);

    let pending = gtk::Box::new(gtk::Orientation::Vertical, 4);
    pending.add_css_class("card");
    pending.set_margin_top(2);
    let head = gtk::Label::new(Some(&format!("{smiles}   \u{2026} querying PrismAI")));
    head.set_halign(gtk::Align::Start);
    head.set_wrap(true);
    pending.append(&head);
    ui.results.prepend(&pending);

    let cmd = PrismCommand::physics_query(&smiles, 5);
    let acc = Rc::new(RefCell::new(String::new()));
    let ui = ui.clone();
    let pending_c = pending.clone();
    let jobs = ui.ctx.jobs.clone();
    let resolver = ui.ctx.resolver.clone();
    jobs.spawn(cmd, &resolver, move |ev| match ev {
        JobEvent::Stdout(line) => acc.borrow_mut().push_str(&line),
        JobEvent::Stderr(_) => {}
        JobEvent::Done(outcome) => {
            ui.results.remove(&pending_c);
            let raw = acc.borrow().clone();
            match serde_json::from_str::<Value>(raw.trim()) {
                Ok(json) if json.get("error").and_then(|e| e.as_str()).is_none() => {
                    let card = build_card(&ui.ctx, &smiles, &json, provenance.as_deref());
                    ui.results.prepend(&card);
                    ui.records.borrow_mut().push(MolRecord::from_json(
                        &smiles,
                        provenance.clone(),
                        &json,
                    ));
                    update_compare_label(&ui);
                    rebuild_compare(&ui);
                    // Only signal success on the actual success path.
                    ui.ctx.toast("Physics computed");
                }
                Ok(json) => {
                    let msg = json
                        .get("error")
                        .and_then(|e| e.as_str())
                        .unwrap_or("PrismAI returned an error");
                    ui.results.prepend(&error_card(&smiles, msg));
                    ui.ctx.toast(msg);
                }
                Err(_) => {
                    let msg = if outcome.success {
                        "PrismAI returned no parseable physics"
                    } else {
                        "PrismAI (:8711) query failed — is the physics model up?"
                    };
                    ui.results.prepend(&error_card(&smiles, msg));
                    ui.ctx.toast(msg);
                }
            }
            ui.predict_btn.set_sensitive(true);
        }
    });
}

fn f64_of(v: &Value, path: &[&str]) -> Option<f64> {
    let mut cur = v;
    for k in path {
        cur = cur.get(k)?;
    }
    cur.as_f64()
}

/// Extract up to five `(field_code, prob)` next-state entries — the REAL basis
/// of the entropy / confidence numbers.
fn top_k_dist(json: &Value) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    let Some(arr) = json.get("top_k_field_codes").and_then(|v| v.as_array()) else {
        return out;
    };
    for el in arr.iter().take(5) {
        let (code, prob) = if let Some(o) = el.as_object() {
            let code = o
                .get("field_code")
                .or_else(|| o.get("code"))
                .map(|c| c.to_string())
                .unwrap_or_default();
            let prob = o
                .get("prob")
                .or_else(|| o.get("probability"))
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0);
            (code, prob)
        } else if let Some(a) = el.as_array() {
            let code = a.first().map(|c| c.to_string()).unwrap_or_default();
            let prob = a.get(1).and_then(|x| x.as_f64()).unwrap_or(0.0);
            (code, prob)
        } else {
            (el.to_string(), 0.0)
        };
        out.push((code, prob));
    }
    out
}

/// Build the field-gated plain-language interpretation. Each sentence is tied to
/// exactly one REAL field and emitted only if that field is present.
fn build_prose(json: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();

    if let Some(conf) = f64_of(json, &["confidence"]) {
        parts.push(format!(
            "The model puts {:.0}% of its weight on a single next physical state (confidence {conf:.2}).",
            conf * 100.0
        ));
    }
    if let Some(cert) = f64_of(json, &["latent_scores", "certainty"]) {
        let (lock, commit) = if cert >= 0.35 {
            ("locked", "well")
        } else if cert >= 0.2 {
            ("moderately locked", "partly")
        } else {
            ("loosely held", "barely")
        };
        parts.push(format!(
            "Thermodynamic certainty {cert:.2} — {lock} — the pocket state is {commit} committed \
             independent of the top prediction."
        ));
    }
    if let Some(ent) = f64_of(json, &["latent_scores", "entropy_nats"]) {
        let quality = if ent < 2.2 {
            "well-defined"
        } else if ent <= 3.2 {
            "somewhat spread"
        } else {
            "ambiguous"
        };
        parts.push(format!(
            "Entropy {ent:.2} nats indicates a {quality} distribution over next states \
             (lower = sharper; ~2.0 well-defined, ~3.8 fully ambiguous)."
        ));
    }
    if let Some(t) = f64_of(json, &["decoded", "time_ps_estimate"])
        .or_else(|| f64_of(json, &["decoded", "time_ps"]))
    {
        parts.push(format!(
            "Predicted kinetic transition ~{:.0} fs — a sub-picosecond, fast local motion.",
            t * 1000.0
        ));
    }
    parts.join(" ")
}

fn build_card(
    ctx: &AppContext,
    smiles: &str,
    json: &Value,
    provenance: Option<&str>,
) -> gtk::Widget {
    if let Some(err) = json.get("error").and_then(|e| e.as_str()) {
        return error_card(smiles, err);
    }
    let card = gtk::Box::new(gtk::Orientation::Vertical, 6);
    card.add_css_class("card");
    card.set_margin_top(2);

    let header = gtk::Label::new(Some(smiles));
    header.add_css_class("mono");
    header.set_halign(gtk::Align::Start);
    header.set_wrap(true);
    header.set_selectable(true);
    card.append(&header);

    if let Some(p) = provenance {
        let chip = gtk::Label::new(Some(p));
        chip.add_css_class("prov-chip");
        chip.add_css_class("prov-derived");
        chip.set_halign(gtk::Align::Start);
        chip.set_wrap(true);
        chip.set_xalign(0.0);
        card.append(&chip);
    }

    // --- numeric readout (unchanged real fields) ---
    let conf = f64_of(json, &["confidence"]).unwrap_or(0.0);
    let cert = f64_of(json, &["latent_scores", "certainty"]).unwrap_or(0.0);
    let ent = f64_of(json, &["latent_scores", "entropy_nats"]);
    let maxp = f64_of(json, &["latent_scores", "max_prob"]);
    card.append(&bar_row("Confidence (top-state prob)", conf, 1.0));
    card.append(&bar_row("Thermodynamic certainty", cert, 1.0));
    if let Some(m) = maxp {
        card.append(&bar_row("Max probability", m, 1.0));
    }
    if let Some(e) = ent {
        card.append(&text_row(
            "Entropy",
            &format!("{e:.3} nats  (lower = better-defined state)"),
        ));
    }
    if let Some(t) = f64_of(json, &["decoded", "time_ps_estimate"])
        .or_else(|| f64_of(json, &["decoded", "time_ps"]))
    {
        card.append(&text_row("Kinetic timescale", &format!("{t:.3} ps")));
    }
    if let Some(code) = json.get("predicted_next_field_code") {
        card.append(&text_row(
            "Predicted next physical state",
            &format!("field code {code}"),
        ));
    }

    // --- (a) next-state distribution: the REAL basis of the entropy number ---
    let dist = top_k_dist(json);
    if !dist.is_empty() {
        let dh = gtk::Label::new(Some("Next-state distribution (top-k field codes)"));
        dh.add_css_class("copilot-tools");
        dh.set_halign(gtk::Align::Start);
        dh.set_margin_top(4);
        card.append(&dh);
        for (code, prob) in &dist {
            let row = bar_row(&format!("field code {code}"), *prob, 1.0);
            row.add_css_class("dist-bar");
            card.append(&row);
        }
        let note = gtk::Label::new(Some(
            "One tall bar = a single dominant next state; a flat spread = many competing states.",
        ));
        note.add_css_class("dim");
        note.set_halign(gtk::Align::Start);
        note.set_wrap(true);
        card.append(&note);
    }

    // --- (b) physics context depth ---
    if let Some(ctx_len) = json.get("context_len").and_then(|x| x.as_i64()) {
        card.append(&text_row(
            "Physics context depth",
            &format!(
                "{ctx_len} tokens (richer = more decoded physics constraints; CCO\u{2248}3, \
                 aspirin\u{2248}21, drug-like indole\u{2248}34)"
            ),
        ));
    }
    if let Some(lat) = json.get("latency_ms").and_then(|x| x.as_f64()) {
        let foot = gtk::Label::new(Some(&format!("server latency {lat:.0} ms")));
        foot.add_css_class("dim");
        foot.set_halign(gtk::Align::Start);
        card.append(&foot);
    }

    // --- per-mechanism channels (null-guard preserved byte-for-byte) ---
    let chans = gtk::Label::new(Some("Per-mechanism channels"));
    chans.add_css_class("copilot-tools");
    chans.set_halign(gtk::Align::Start);
    chans.set_margin_top(4);
    card.append(&chans);
    let mut any_family_inactive = false;
    for (name, pretty) in [
        ("observation", "observation (spike/hydration)"),
        ("tsh", "TSH (transition state)"),
        ("epistemic", "epistemic (uncertainty)"),
    ] {
        let fam = json.get("per_family").and_then(|p| p.get(name));
        match fam {
            Some(f) if !f.is_null() => {
                let c = f
                    .get("confidence")
                    .and_then(|x| x.as_f64())
                    .or_else(|| f.get("prob").and_then(|x| x.as_f64()))
                    .unwrap_or(0.0);
                let code = f
                    .get("predicted_field_code")
                    .map(|x| x.to_string())
                    .unwrap_or_default();
                card.append(&text_row(pretty, &format!("code {code}  conf {c:.3}")));
            }
            _ => {
                any_family_inactive = true;
                card.append(&text_row(
                    pretty,
                    "\u{2014}  (not predicted for this molecule)",
                ));
            }
        }
    }
    if any_family_inactive {
        let foot = gtk::Label::new(Some(
            "TSH / epistemic channels are not emitted by the current physics server for this \
             molecule (only the observation channel fires live).",
        ));
        foot.add_css_class("dim");
        foot.set_halign(gtk::Align::Start);
        foot.set_wrap(true);
        card.append(&foot);
    }

    // --- (c) plain-language interpretation ---
    let prose = build_prose(json);
    if !prose.is_empty() {
        let interp = gtk::Label::new(Some(&prose));
        interp.add_css_class("subtitle");
        interp.set_halign(gtk::Align::Start);
        interp.set_wrap(true);
        interp.set_xalign(0.0);
        interp.set_margin_top(4);
        interp.set_selectable(true);
        card.append(&interp);
    }

    // --- Save interpretation button ---
    let save_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    save_row.set_halign(gtk::Align::End);
    let save_btn = gtk::Button::with_label("Save interpretation\u{2026}");
    save_row.append(&save_btn);
    card.append(&save_row);
    {
        let ctx = ctx.clone();
        let md = interpretation_markdown(smiles, json, provenance, &prose);
        let smiles_name = sanitize_filename(smiles);
        save_btn.connect_clicked(move |_| {
            let dialog = gtk::FileDialog::new();
            dialog.set_title("Save PrismAI interpretation");
            dialog.set_initial_name(Some(&format!("prism_physics_{smiles_name}.md")));
            let ctx = ctx.clone();
            let md = md.clone();
            let window = ctx.window.clone();
            dialog.save(Some(&window), gtk::gio::Cancellable::NONE, move |result| {
                if let Ok(file) = result {
                    if let Some(path) = file.path() {
                        match std::fs::write(&path, &md) {
                            Ok(_) => ctx.toast(&format!("Saved {}", path.display())),
                            Err(e) => ctx.toast(&format!("Save failed: {e}")),
                        }
                    }
                }
            });
        });
    }

    card.upcast()
}

/// Assemble the exportable markdown (prose + numeric table).
fn interpretation_markdown(
    smiles: &str,
    json: &Value,
    provenance: Option<&str>,
    prose: &str,
) -> String {
    let mut out = String::new();
    out.push_str("# PrismAI physics interpretation\n\n");
    out.push_str(&format!("**SMILES:** `{smiles}`\n\n"));
    if let Some(p) = provenance {
        out.push_str(&format!("**Provenance:** {p}\n\n"));
    }
    if !prose.is_empty() {
        out.push_str(prose);
        out.push_str("\n\n");
    }
    out.push_str("| Field | Value |\n|---|---|\n");
    let mut row = |label: &str, val: String| out.push_str(&format!("| {label} | {val} |\n"));
    if let Some(v) = f64_of(json, &["confidence"]) {
        row("Confidence (top-state prob)", format!("{v:.4}"));
    }
    if let Some(v) = f64_of(json, &["latent_scores", "certainty"]) {
        row("Thermodynamic certainty", format!("{v:.4}"));
    }
    if let Some(v) = f64_of(json, &["latent_scores", "entropy_nats"]) {
        row("Entropy (nats)", format!("{v:.4}"));
    }
    if let Some(v) = f64_of(json, &["latent_scores", "max_prob"]) {
        row("Max probability", format!("{v:.4}"));
    }
    if let Some(v) = f64_of(json, &["decoded", "time_ps_estimate"])
        .or_else(|| f64_of(json, &["decoded", "time_ps"]))
    {
        row("Kinetic timescale (ps)", format!("{v:.4}"));
        row("Kinetic timescale (fs)", format!("{:.0}", v * 1000.0));
    }
    if let Some(v) = json.get("context_len").and_then(|x| x.as_i64()) {
        row("Physics context depth (tokens)", v.to_string());
    }
    out.push_str(
        "\n_Advisory / physics only. TSH & epistemic per-family channels are null on the live \
         server for every molecule; only the observation channel fires._\n",
    );
    out
}

fn sanitize_filename(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    cleaned.chars().take(32).collect()
}

fn error_card(smiles: &str, msg: &str) -> gtk::Widget {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 4);
    card.add_css_class("card");
    let h = gtk::Label::new(Some(smiles));
    h.add_css_class("mono");
    h.set_halign(gtk::Align::Start);
    h.set_wrap(true);
    let e = gtk::Label::new(Some(&format!("\u{26A0} {msg}")));
    e.set_halign(gtk::Align::Start);
    e.set_wrap(true);
    card.append(&h);
    card.append(&e);
    card.upcast()
}

fn bar_row(label: &str, value: f64, max: f64) -> gtk::Box {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let l = gtk::Label::new(Some(label));
    l.set_width_chars(26);
    l.set_xalign(0.0);
    let bar = gtk::LevelBar::new();
    bar.set_min_value(0.0);
    bar.set_max_value(max);
    bar.set_value(value.clamp(0.0, max));
    bar.set_hexpand(true);
    bar.set_valign(gtk::Align::Center);
    let v = gtk::Label::new(Some(&format!("{value:.3}")));
    v.set_width_chars(6);
    v.set_xalign(1.0);
    v.add_css_class("mono");
    row.append(&l);
    row.append(&bar);
    row.append(&v);
    row
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
    row.append(&l);
    row.append(&v);
    row
}

// ------------------------------------------------------------------------
// Multi-molecule compare grid.
// ------------------------------------------------------------------------

fn update_compare_label(ui: &Rc<Oracle>) {
    let n = ui.records.borrow().len();
    ui.compare_toggle.set_label(&format!("Compare ({n})"));
}

fn rebuild_compare(ui: &Rc<Oracle>) {
    while let Some(child) = ui.compare_container.first_child() {
        ui.compare_container.remove(&child);
    }
    let recs = ui.records.borrow();
    if !ui.compare_toggle.is_active() || recs.len() < 2 {
        return;
    }
    let card = build_compare_card(ui, &recs);
    ui.compare_container.append(&card);
}

fn build_compare_card(ui: &Rc<Oracle>, recs: &[MolRecord]) -> gtk::Widget {
    let n = recs.len();

    // Per-column rank indices over REAL returned fields (no composite score).
    let conf: Vec<f64> = recs.iter().map(|r| r.confidence).collect();
    let cert: Vec<f64> = recs.iter().map(|r| r.certainty).collect();
    let ent: Vec<Option<f64>> = recs.iter().map(|r| r.entropy_nats).collect();
    let ctxf: Vec<Option<f64>> = recs
        .iter()
        .map(|r| r.context_len.map(|x| x as f64))
        .collect();

    let best_conf = argmax_f(&conf);
    let worst_conf = argmin_f(&conf);
    let best_cert = argmax_f(&cert);
    let worst_cert = argmin_f(&cert);
    let best_ent = argmin_opt(&ent); // sharpest = LOWEST entropy
    let worst_ent = argmax_opt(&ent);
    let best_ctx = argmax_opt(&ctxf); // richest = most context
    let worst_ctx = argmin_opt(&ctxf);

    let card = gtk::Box::new(gtk::Orientation::Vertical, 6);
    card.add_css_class("card");
    let heading = gtk::Label::new(Some(&format!("Compare — {n} molecules")));
    heading.add_css_class("subtitle");
    heading.set_halign(gtk::Align::Start);
    card.append(&heading);

    let grid = gtk::Grid::new();
    grid.add_css_class("compare-grid");
    grid.set_row_spacing(2);
    grid.set_column_spacing(6);

    let headers = [
        "Molecule",
        "Confidence",
        "Certainty",
        "Entropy (nats)",
        "Kinetic (fs)",
        "Context depth",
    ];
    for (c, h) in headers.iter().enumerate() {
        let l = gtk::Label::new(Some(h));
        l.add_css_class("subtitle");
        l.set_xalign(0.0);
        grid.attach(&l, c as i32, 0, 1, 1);
    }

    for (i, r) in recs.iter().enumerate() {
        let row = (i + 1) as i32;

        let sl = gtk::Label::new(Some(&truncate(&r.smiles, 28)));
        sl.add_css_class("mono");
        sl.set_xalign(0.0);
        sl.set_tooltip_text(Some(&r.smiles));
        grid.attach(&sl, 0, row, 1, 1);

        let c1 = value_cell(&format!("{:.3}", r.confidence));
        apply_rank(&c1, i, best_conf, worst_conf);
        grid.attach(&c1, 1, row, 1, 1);

        let c2 = value_cell(&format!("{:.3}", r.certainty));
        apply_rank(&c2, i, best_cert, worst_cert);
        grid.attach(&c2, 2, row, 1, 1);

        let c3 = value_cell(&opt_fmt(r.entropy_nats, 3));
        apply_rank(&c3, i, best_ent, worst_ent);
        grid.attach(&c3, 3, row, 1, 1);

        let c4 = value_cell(&opt_fmt(r.time_ps.map(|t| t * 1000.0), 0));
        grid.attach(&c4, 4, row, 1, 1);

        let c5 = value_cell(
            &r.context_len
                .map(|x| x.to_string())
                .unwrap_or_else(|| "\u{2014}".to_string()),
        );
        apply_rank(&c5, i, best_ctx, worst_ctx);
        grid.attach(&c5, 5, row, 1, 1);

        let drop = gtk::Button::with_label("\u{2715}");
        drop.add_css_class("flat");
        drop.set_tooltip_text(Some("Remove this molecule from the comparison"));
        let ui2 = ui.clone();
        drop.connect_clicked(move |_| {
            {
                let mut recs = ui2.records.borrow_mut();
                if i < recs.len() {
                    recs.remove(i);
                }
            }
            update_compare_label(&ui2);
            rebuild_compare(&ui2);
        });
        grid.attach(&drop, 6, row, 1, 1);
    }
    card.append(&grid);

    // plain-language footer, keyed to the most thermodynamically-locked molecule
    if let Some(bi) = best_cert {
        let r = &recs[bi];
        let ent_str = r
            .entropy_nats
            .map(|e| format!("{e:.2}"))
            .unwrap_or_else(|| "n/a".to_string());
        let footer = gtk::Label::new(Some(&format!(
            "Of {n} molecules, {} shows the sharpest, most thermodynamically-locked next state \
             (certainty {:.2}, entropy {ent_str}).",
            truncate(&r.smiles, 40),
            r.certainty
        )));
        footer.add_css_class("subtitle");
        footer.set_halign(gtk::Align::Start);
        footer.set_wrap(true);
        footer.set_xalign(0.0);
        card.append(&footer);
    }

    card.upcast()
}

fn value_cell(text: &str) -> gtk::Label {
    let l = gtk::Label::new(Some(text));
    l.add_css_class("mono");
    l.set_xalign(0.0);
    l
}

fn apply_rank(cell: &gtk::Label, i: usize, best: Option<usize>, worst: Option<usize>) {
    if best == Some(i) {
        cell.add_css_class("compare-best");
    } else if worst == Some(i) {
        cell.add_css_class("compare-worst");
    }
}

fn opt_fmt(v: Option<f64>, decimals: usize) -> String {
    match v {
        Some(v) => format!("{v:.decimals$}"),
        None => "\u{2014}".to_string(),
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() > n {
        let head: String = s.chars().take(n.saturating_sub(1)).collect();
        format!("{head}\u{2026}")
    } else {
        s.to_string()
    }
}

fn argmax_f(v: &[f64]) -> Option<usize> {
    v.iter()
        .enumerate()
        .fold(None, |acc, (i, &x)| match acc {
            None => Some((i, x)),
            Some((_, best)) if x > best => Some((i, x)),
            other => other,
        })
        .map(|(i, _)| i)
}

fn argmin_f(v: &[f64]) -> Option<usize> {
    v.iter()
        .enumerate()
        .fold(None, |acc, (i, &x)| match acc {
            None => Some((i, x)),
            Some((_, best)) if x < best => Some((i, x)),
            other => other,
        })
        .map(|(i, _)| i)
}

fn argmax_opt(v: &[Option<f64>]) -> Option<usize> {
    v.iter()
        .enumerate()
        .filter_map(|(i, x)| x.map(|x| (i, x)))
        .fold(None, |acc, (i, x)| match acc {
            None => Some((i, x)),
            Some((_, best)) if x > best => Some((i, x)),
            other => other,
        })
        .map(|(i, _)| i)
}

fn argmin_opt(v: &[Option<f64>]) -> Option<usize> {
    v.iter()
        .enumerate()
        .filter_map(|(i, x)| x.map(|x| (i, x)))
        .fold(None, |acc, (i, x)| match acc {
            None => Some((i, x)),
            Some((_, best)) if x < best => Some((i, x)),
            other => other,
        })
        .map(|(i, _)| i)
}
