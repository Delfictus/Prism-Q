//! Run Engine — launch a canonical PRISM-4D MD run from the desktop.
//!
//! The operator picks a topology and output directory, sees the EXACT canonical
//! run command (typed preset or complete governed argv, rendered from
//! [`PrismCommand::validate_and_run`]), clicks **Launch**, and watches a live
//! streaming log with preflight / engine / postflight gate chips. The engine is
//! launched through the governed `scripts/prism-validate-and-run.sh` adapter,
//! which forwards the complete argv after preflight. A **Cancel** button
//! group-kills the whole process tree (adapter + engine).
//!
//! Success is fail-closed: the governed adapter must exit zero and its selected
//! postflight must pass. Typed presets and complete-argv runs that request
//! `--md-only-evidence` additionally require `md_evidence_manifest.json` with
//! `total_spikes_md > 0`. Only then is the app Run Root pointed at the output;
//! a cancelled run never sets it.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;

use adw::prelude::*;

use crate::adapters::{EngineArgv, EngineSurface, PrismBin, PrismCommand, RunParams, SurfaceFlag};
use crate::app::AppContext;
use crate::jobs::{JobEvent, JobHandle};

/// Residue count above which the sizing rule calls for `--multi-stream 20`
/// (script header line 17: `>400: --multi-stream 20`, otherwise 8).
const MULTI_STREAM_LARGE_THRESHOLD: i64 = 400;

/// Command-class presets (CLAUDE.md doc presets — NOT an engine enum). Each maps
/// to a distinct `PrismCommand` constructor so the preview == what fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Preset {
    /// C-canonical — calibrated base + typed surface tail (`validate_and_run_ext`).
    Canonical,
    /// C-twin-graph — `--coupled-twin --graph-coupling` substituted for
    /// `--multi-differential` (`validate_and_run_twin`).
    TwinGraph,
    /// C-smoke-min — `--fast-25k --multi-stream 1` (`smoke_min`). Not production.
    SmokeMin,
    /// C-smoke-eng — `--fast-25k --multi-stream 2` (`smoke_eng`). Not production.
    SmokeEng,
    /// Complete live engine argv. Topology/output stay typed; every additional
    /// token comes from the one-token-per-line governed editor.
    CompleteArgv,
}

/// The multichain "decision" from CLAUDE.md's Multichain procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultichainDecision {
    /// (a) interface/cryptic site at a chain interface → merge chains.
    Merge,
    /// (b) chains independent → single-chain, one per run.
    Independent,
    /// (c) one chain is ligand/peptide/cofactor → keep the protein chain only.
    LigandProtein,
}

/// The staged, gated prep state machine. Each phase is one operator-fired,
/// gated job; the app NEVER auto-advances between phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrepPhase {
    /// No raw PDB loaded (or a topology was chosen directly, skipping prep).
    Idle,
    /// Raw PDB loaded; clean phase can be fired.
    NeedClean,
    /// Clean passed; a merge is required (multichain Merge decision).
    NeedMerge,
    /// Clean (and any merge) passed; prep-topology can be fired.
    NeedPrep,
    /// Prep passed; a topology was produced and funneled into `set_topology`.
    Ready,
}

struct RunEngine {
    ctx: AppContext,
    primary_manifest: Option<PathBuf>,
    primary_panel: RefCell<Option<PathBuf>>,
    primary_panel_verified: Cell<bool>,
    primary_panel_label: gtk::Label,
    primary_results: gtk::Box,
    // --- selection + config state ---
    topo: RefCell<Option<PathBuf>>,
    output: RefCell<Option<PathBuf>>,
    multi_stream: Cell<u32>,
    // --- run-mode: single debug run vs N-replicate consensus (production) ---
    // CLAUDE.md: "Single runs are for debugging. Production = replicate consensus."
    // In consensus mode the panel drives scripts/prism_replicate.py instead of the
    // single-run wrapper; every other single-run control is left untouched.
    consensus_mode: Cell<bool>,
    n_replicates: Cell<usize>,
    // --- PDB intake + prep state ---
    raw_pdb: RefCell<Option<PathBuf>>,
    chains: RefCell<Vec<String>>,
    cleaned_pdb: RefCell<Option<PathBuf>>,
    merged_pdb: RefCell<Option<PathBuf>>,
    chain_map: RefCell<Option<PathBuf>>,
    prep_phase: Cell<PrepPhase>,
    decision: Cell<Option<MultichainDecision>>,
    prep_clean_ok: Cell<bool>,
    prep_prep_ok: Cell<bool>,
    prep_merge_ok: Cell<bool>,
    // --- surface / mode state ---
    preset: Cell<Preset>,
    /// First-class NAMED engine surface. When != `None` it OVERRIDES the preset
    /// + advanced-surface controls and emits `--engine-surface <value>` directly.
    engine_surface: Cell<EngineSurface>,
    max_wall: Cell<u64>,
    spike_percentile: Cell<u32>,
    rest2: Cell<bool>,
    multi_temp: Cell<bool>,
    stepped_holds: Cell<bool>,
    clustering_backend: RefCell<Option<String>>,
    ghost_so3: Cell<f32>,
    ghost_cell_a: Cell<f32>,
    ghost_max_temporal: Cell<u64>,
    ghost_step_bucket: Cell<u64>,
    // --- run-parameter tabs (protocol-aware; threaded into ext + named-surface paths) ---
    // `standard_protocol` selects the STANDARD (user-configurable thermodynamics)
    // protocol: the base then drops --fast + --multi-differential and the engine
    // honors --temperature/--cryo-temp/--steps 1:1 (nhs_rt_full.rs:5094-5112
    // start_temp/end_temp, :9080 steps_per_stream = args.steps). Under the default
    // FAST protocol those three are fixed by calibration, so temperature/cryo are not
    // emitted and --steps is emitted only when it differs from the 500000 default
    // (then paired with --wall-bound-hysteresis). --cluster-threshold is never
    // emitted (the engine never consumes it).
    standard_protocol: Cell<bool>,
    temperature: Cell<f32>,
    cryo_temp: Cell<f32>,
    steps: Cell<i32>,
    save_traj_interval: Cell<u32>,
    lining_cutoff: Cell<f32>,
    // --- complete live engine argv (one exact token per non-empty line) ---
    engine_argv: RefCell<EngineArgv>,
    engine_argv_error: RefCell<Option<String>>,
    // Frozen at launch so editing controls mid-stream cannot weaken acceptance.
    active_md_only_required: Cell<bool>,
    active_consensus_required: Cell<bool>,
    // --- run lifecycle state ---
    running: Cell<bool>,
    cancelled: Cell<bool>,
    job: RefCell<Option<JobHandle>>,
    postflight_pass: Cell<Option<bool>>,
    script_ok: Cell<bool>,
    // --- widgets ---
    topo_label: gtk::Label,
    output_label: gtk::Label,
    residue_note: gtk::Label,
    cmd_preview: gtk::Label,
    preflight_chip: gtk::Label,
    engine_chip: gtk::Label,
    postflight_chip: gtk::Label,
    launch_btn: gtk::Button,
    cancel_btn: gtk::Button,
    stream8: gtk::ToggleButton,
    stream20: gtk::ToggleButton,
    log_view: gtk::TextView,
    // --- prep widgets ---
    pdb_label: gtk::Label,
    chain_label: gtk::Label,
    decision_row: gtk::Box,
    clean_chip: gtk::Label,
    merge_chip: gtk::Label,
    prep_chip: gtk::Label,
    clean_btn: gtk::Button,
    merge_btn: gtk::Button,
    prep_btn: gtk::Button,
    advisory_label: gtk::Label,
}

/// The interactive surface-selector widgets, returned by [`build_surface_card`]
/// so their change handlers can be wired to the [`RunEngine`] state after it is
/// constructed.
struct SurfaceWidgets {
    preset: gtk::DropDown,
    engine_surface: gtk::DropDown,
    clustering: gtk::DropDown,
    rest2: gtk::CheckButton,
    multi_temp: gtk::CheckButton,
    stepped_holds: gtk::CheckButton,
    max_wall: gtk::SpinButton,
    spike_percentile: gtk::SpinButton,
    ghost_so3: gtk::SpinButton,
    ghost_cell_a: gtk::SpinButton,
    ghost_max_temporal: gtk::SpinButton,
    ghost_step_bucket: gtk::SpinButton,
    ghost_box: gtk::Box,
    typed_controls: gtk::Box,
}

/// Protocol-selector labels for the Thermodynamics tab, index-aligned: index 0 =
/// Fast (calibrated, default), index 1 = Standard (user-configurable thermodynamics).
/// Standard drops --fast + --multi-differential so the engine honors
/// --temperature/--cryo-temp/--steps 1:1 (nhs_rt_full.rs:5094-5112 / :9080).
const PROTOCOL_LABELS: [&str; 2] = [
    "Fast-35K (calibrated, default)",
    "Standard (configurable thermodynamics)",
];

/// Run-mode labels, index-aligned: 0 = single debug run (the single-run wrapper),
/// 1 = N-replicate consensus (scripts/prism_replicate.py, the CLAUDE.md production
/// entry point).
const RUN_MODE_LABELS: [&str; 2] = ["Single run (debug)", "N-replicate consensus (production)"];

/// Command-class preset labels, index-aligned with [`preset_from_index`].
const PRESET_LABELS: [&str; 5] = [
    "C-canonical (production)",
    "C-twin-graph (coupled TWIN, captured graph)",
    "C-smoke-min (init check \u{2014} not production)",
    "C-smoke-eng (stream exercise \u{2014} not production)",
    "Complete engine argv (all nhs_rt_full options/modes/stages)",
];

fn preset_from_index(i: u32) -> Preset {
    match i {
        1 => Preset::TwinGraph,
        2 => Preset::SmokeMin,
        3 => Preset::SmokeEng,
        4 => Preset::CompleteArgv,
        _ => Preset::Canonical,
    }
}

/// First-class NAMED engine-surface choices, index-aligned with
/// [`engine_surface_from_index`]. 1:1 with the engine's `--engine-surface`
/// value_parser (nhs_rt_full.rs:1041). Selecting a named surface emits the single
/// `--engine-surface <value>` token; the engine expands it upstream of every
/// guard/dispatch (:3666-3699). "always-on sensor" / a bare interferometric
/// toggle are DELIBERATELY absent — they have no distinct engine flag (:1032-1034).
const ENGINE_SURFACE_LABELS: [&str; 5] = [
    "none  (use the command class above)",
    "captured-graph-twin  (coupled TWIN + captured graph)",
    "interferometric-twin  (multi-differential)",
    "twin16  (multi-differential, streams \u{2265}16)",
    "ghost-tiles  (ghost-phase-lattice-4d + mar-v2 + m1)",
];

fn engine_surface_from_index(i: u32) -> EngineSurface {
    match i {
        1 => EngineSurface::CapturedGraphTwin,
        2 => EngineSurface::InterferometricTwin,
        3 => EngineSurface::Twin16,
        4 => EngineSurface::GhostTiles,
        _ => EngineSurface::None,
    }
}

/// Clustering-backend choices from the engine. Native compatibility checks remain
/// authoritative; for example grid + multi-differential is rejected by the engine.
const CLUSTERING_LABELS: [&str; 6] = [
    "auto",
    "gpu-hash",
    "ghost-phase-lattice-4d",
    "optix",
    "lbvh",
    "grid",
];

/// Build the "Surface & Mode" card. Common controls are typed; the complete argv
/// editor below the card covers the rest of the live `nhs_rt_full` clap surface.
fn build_surface_card() -> (gtk::Box, SurfaceWidgets) {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 8);
    card.add_css_class("card");

    let heading = gtk::Label::new(Some("Surface & Mode"));
    heading.add_css_class("subtitle");
    heading.set_halign(gtk::Align::Start);
    card.append(&heading);

    // Preset dropdown
    let preset_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let preset_lbl = gtk::Label::new(Some("Command class"));
    preset_lbl.set_width_chars(16);
    preset_lbl.set_xalign(0.0);
    let preset = gtk::DropDown::from_strings(&PRESET_LABELS);
    preset.set_selected(0);
    preset_row.append(&preset_lbl);
    preset_row.append(&preset);
    card.append(&preset_row);

    let typed_controls = gtk::Box::new(gtk::Orientation::Vertical, 8);

    // First-class NAMED engine-surface selector. When != "none" it emits the
    // single --engine-surface <value> token and OVERRIDES the command class +
    // advanced surfaces below (the engine expands the named value upstream of
    // every guard/dispatch, nhs_rt_full.rs:3666-3699).
    let es_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let es_lbl = gtk::Label::new(Some("Engine surface"));
    es_lbl.set_width_chars(16);
    es_lbl.set_xalign(0.0);
    let engine_surface = gtk::DropDown::from_strings(&ENGINE_SURFACE_LABELS);
    engine_surface.set_selected(0);
    es_row.append(&es_lbl);
    es_row.append(&engine_surface);
    typed_controls.append(&es_row);
    let es_note = gtk::Label::new(Some(
        "Named engine surface (--engine-surface). When set, it emits ONE flag the engine EXPANDS \
         into the real config and OVERRIDES the command class + advanced surfaces below; the base \
         drops --multi-differential so captured-graph-twin is not silently no-op'd. \"always-on \
         sensor\" and a bare interferometric/sensor toggle are NOT values (no distinct engine flag).",
    ));
    es_note.add_css_class("dim");
    es_note.set_wrap(true);
    es_note.set_xalign(0.0);
    typed_controls.append(&es_note);

    // Calibrated ranker for the typed preset. Complete argv mode can select any
    // ranker admitted by the live engine.
    let ranker = gtk::Label::new(Some(
        "Typed-preset site ranker: phase-manifold  (complete argv mode is fully tunable)",
    ));
    ranker.add_css_class("surface-locked");
    ranker.set_halign(gtk::Align::Start);
    ranker.set_xalign(0.0);
    typed_controls.append(&ranker);

    // Advanced surfaces expander
    let expander = gtk::Expander::new(Some("Advanced surfaces"));
    let adv = gtk::Box::new(gtk::Orientation::Vertical, 6);
    adv.set_margin_top(6);

    // MD stages
    let rest2 = gtk::CheckButton::with_label("Solute Tempering (\u{03BB}-ladder) (--rest2)");
    rest2.set_tooltip_text(Some(
        "Per-stream REST2 \u{03BB}-ladder solute tempering \u{2014} NOT Hamiltonian replica \
         exchange; there are no swap moves (nhs_rt_full.rs:9746).",
    ));
    let multi_temp = gtk::CheckButton::with_label("Multi-temperature ladder (--multi-temp)");
    let stepped_holds = gtk::CheckButton::with_label("Stepped temperature holds (--stepped-holds)");
    adv.append(&rest2);
    adv.append(&multi_temp);
    adv.append(&stepped_holds);

    // Max wall seconds
    let mw_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let mw_lbl = gtk::Label::new(Some("Max wall seconds (--path-a-max-wall-seconds)"));
    mw_lbl.set_xalign(0.0);
    let max_wall = gtk::SpinButton::with_range(30.0, 3600.0, 10.0);
    max_wall.set_value(180.0);
    mw_row.append(&mw_lbl);
    mw_row.append(&max_wall);
    adv.append(&mw_row);

    // Spike percentile
    let sp_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let sp_lbl = gtk::Label::new(Some("Spike percentile (--spike-percentile)"));
    sp_lbl.set_xalign(0.0);
    let spike_percentile = gtk::SpinButton::with_range(1.0, 99.0, 1.0);
    spike_percentile.set_value(70.0);
    sp_row.append(&sp_lbl);
    sp_row.append(&spike_percentile);
    adv.append(&sp_row);

    // Clustering backend
    let cb_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let cb_lbl = gtk::Label::new(Some("Clustering backend (--clustering-backend)"));
    cb_lbl.set_xalign(0.0);
    let clustering = gtk::DropDown::from_strings(&CLUSTERING_LABELS);
    clustering.set_selected(0);
    cb_row.append(&cb_lbl);
    cb_row.append(&clustering);
    adv.append(&cb_row);
    let cb_note = gtk::Label::new(Some(
        "All engine backends are shown. The engine rejects incompatible combinations such as \
         grid + --multi-differential. ghost-phase-lattice-4d also adds \
         --m1-monolithic-discovery + --mar-v2-telemetry in this typed preset.",
    ));
    cb_note.add_css_class("dim");
    cb_note.set_wrap(true);
    cb_note.set_xalign(0.0);
    adv.append(&cb_note);

    // Ghost-phase-lattice-4d parameters (only emitted when that backend chosen)
    let ghost_box = gtk::Box::new(gtk::Orientation::Vertical, 4);
    let ghost_so3 = labeled_spin(
        &ghost_box,
        "SO(3) threshold (--ghost-phase-lattice-so3-threshold)",
        0.0,
        1.0,
        0.05,
        0.75,
    );
    let ghost_cell_a = labeled_spin(
        &ghost_box,
        "Cell edge \u{00c5} (--ghost-phase-lattice-cell-a)",
        1.0,
        50.0,
        0.5,
        5.0,
    );
    let ghost_max_temporal = labeled_spin(
        &ghost_box,
        "Max temporal edge steps (--ghost-phase-lattice-max-temporal-edge-steps)",
        1.0,
        100000.0,
        50.0,
        500.0,
    );
    let ghost_step_bucket = labeled_spin(
        &ghost_box,
        "Step bucket size (--ghost-phase-lattice-step-bucket-size)",
        1.0,
        100000.0,
        50.0,
        500.0,
    );
    ghost_box.set_sensitive(false); // enabled only for the ghost-lattice-4d backend
    adv.append(&ghost_box);

    // Calibrated body-unroll for the typed preset.
    let unroll = gtk::Label::new(Some(
        "Typed-preset captured-graph body unroll: 1 (complete argv mode is tunable)",
    ));
    unroll.add_css_class("surface-locked");
    unroll.set_xalign(0.0);
    adv.append(&unroll);

    // "Not exposed by the current engine CLI" info rows
    let na = gtk::Label::new(Some(
        "captured-graph-twin / interferometric-twin / twin16 / ghost-tiles are now FIRST-CLASS via \
         the Engine surface dropdown above (each emits one --engine-surface value the engine \
         expands). Still NOT flags (no clap field \u{2014} not fabricated): \"always-on sensor\" \
         (unconditional baseline spike writer), a bare standalone interferometric/sensor toggle, and \
         literal GhostTileRing / GhostTileFrame. The manual controls here remain for a hand-tuned \
         Canonical run (e.g. custom ghost_phase_lattice_* knobs) when no named surface is selected. \
         Complete argv mode exposes every additional engine option, including persistent-coupling; \
         engine-native hardware/compatibility guards still decide whether a combination can run.",
    ));
    na.add_css_class("surface-unavailable");
    na.set_wrap(true);
    na.set_xalign(0.0);
    adv.append(&na);

    expander.set_child(Some(&adv));
    typed_controls.append(&expander);
    card.append(&typed_controls);

    (
        card,
        SurfaceWidgets {
            preset,
            engine_surface,
            clustering,
            rest2,
            multi_temp,
            stepped_holds,
            max_wall,
            spike_percentile,
            ghost_so3,
            ghost_cell_a,
            ghost_max_temporal,
            ghost_step_bucket,
            ghost_box,
            typed_controls,
        },
    )
}

struct CompleteArgvWidgets {
    card: gtk::Box,
    editor: gtk::TextView,
    status: gtk::Label,
}

/// Build the lossless advanced argv editor. One non-empty line is one exact
/// process argument; execution still uses `Command::args` and never a shell.
fn build_complete_argv_card() -> CompleteArgvWidgets {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 8);
    card.add_css_class("card");
    card.set_visible(false);

    let heading = gtk::Label::new(Some("Complete nhs_rt_full argv"));
    heading.add_css_class("subtitle");
    heading.set_halign(gtk::Align::Start);
    card.append(&heading);

    let note = gtk::Label::new(Some(
        "One exact argv token per non-empty line. Spaces inside a line remain inside that token; \
         no shell is used. Topology and output are selected above and cannot be repeated here. \
         Every other current or future engine option, value, mode, stage, or subcommand is \
         forwarded in order. Run `prismos-engine-capabilities --pretty` for the installed \
         engine inventory; default-on optional booleans can be disabled with an exact \
         `--flag=false` token. Deprecated/legacy selectors (including xgb-v3 and tokenized-v4) \
         remain reproducibility-only; reachability does not promote them. Engine-native clap \
         and runtime conflict gates remain authoritative. Adapter-only packet/bundle/coverage \
         controls are identified separately by the inventory and become governed environment \
         bindings rather than invalid engine clap tokens.",
    ));
    note.add_css_class("dim");
    note.set_wrap(true);
    note.set_xalign(0.0);
    card.append(&note);

    let editor = gtk::TextView::new();
    editor.set_monospace(true);
    editor.set_wrap_mode(gtk::WrapMode::None);
    editor.set_top_margin(6);
    editor.set_bottom_margin(6);
    editor.set_left_margin(6);
    editor.set_right_margin(6);
    editor.add_css_class("mono");
    let scroll = gtk::ScrolledWindow::builder()
        .min_content_height(180)
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&editor)
        .build();
    card.append(&scroll);

    let status = gtk::Label::new(Some("0 additional engine argv tokens"));
    status.add_css_class("dim");
    status.set_halign(gtk::Align::Start);
    status.set_xalign(0.0);
    status.set_wrap(true);
    card.append(&status);

    CompleteArgvWidgets {
        card,
        editor,
        status,
    }
}

/// Append a `label + SpinButton` row to `parent`; return the spin.
fn labeled_spin(
    parent: &gtk::Box,
    label: &str,
    min: f64,
    max: f64,
    step: f64,
    value: f64,
) -> gtk::SpinButton {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let lbl = gtk::Label::new(Some(label));
    lbl.set_hexpand(true);
    lbl.set_xalign(0.0);
    lbl.set_wrap(true);
    let spin = gtk::SpinButton::with_range(min, max, step);
    spin.set_value(value);
    if step < 1.0 {
        spin.set_digits(2);
    }
    row.append(&lbl);
    row.append(&spin);
    parent.append(&row);
    spin
}

/// The physics / capture / detection widgets, returned by [`build_params_tabs`] so
/// their change handlers can be wired to [`RunEngine`] after it is constructed.
/// These are FUNDAMENTAL engine knobs and apply to EVERY run path — a named engine
/// surface never greys them.
struct ParamWidgets {
    /// Protocol selector: Fast-35K (calibrated, default) vs Standard (configurable
    /// thermodynamics). On "Standard" the handler un-greys temperature/cryo and sets
    /// `standard_protocol = true`; the STANDARD engine protocol honors the setpoints.
    protocol: gtk::DropDown,
    /// `--temperature` / `--cryo-temp` spins. Live only in Standard mode (greyed
    /// under Fast, where the calibrated protocol fixes the thermal cycle and the
    /// engine does not read them). Their handlers set the matching cells.
    temperature: gtk::SpinButton,
    cryo_temp: gtk::SpinButton,
    /// `--steps` spin — ALWAYS live. In Standard mode the engine honors it directly;
    /// in Fast mode a non-default value is paired with --wall-bound-hysteresis.
    steps: gtk::SpinButton,
    save_trajectory_interval: gtk::SpinButton,
    lining_cutoff: gtk::SpinButton,
}

/// Build the three-tab (`gtk::Notebook`) physics/capture/detection parameter
/// surface. Every spin maps 1:1 to a VERIFIED `nhs_rt_full` clap flag and is
/// threaded into BOTH the Canonical (ext) and named-engine-surface run paths, so
/// these tabs are FUNDAMENTAL, not surface-overridden.
fn build_params_tabs() -> (gtk::Notebook, ParamWidgets) {
    let notebook = gtk::Notebook::new();
    notebook.add_css_class("card");

    // Page 1 — Thermodynamics. A Protocol selector chooses between the calibrated
    // Fast-35K protocol (default) and the STANDARD protocol. Under Fast the engine
    // FIXES the thermal cycle (50 K -> 310 K) and the step budget, so the
    // temperature/cryo setpoints are greyed and not emitted. Under Standard the run
    // drops --fast + --multi-differential and the engine honors --temperature/
    // --cryo-temp/--steps 1:1 (nhs_rt_full.rs:5094-5112 start_temp/end_temp, :9080
    // steps_per_stream = args.steps), so those spins go live. --steps is ALWAYS live
    // (in Fast a non-default value is paired with --wall-bound-hysteresis).
    let thermo = gtk::Box::new(gtk::Orientation::Vertical, 4);
    thermo.set_margin_top(6);
    thermo.set_margin_bottom(6);
    thermo.set_margin_start(6);
    thermo.set_margin_end(6);

    // Protocol selector (Fast vs Standard).
    let proto_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let proto_lbl = gtk::Label::new(Some("Protocol"));
    proto_lbl.set_width_chars(16);
    proto_lbl.set_xalign(0.0);
    let protocol = gtk::DropDown::from_strings(&PROTOCOL_LABELS);
    protocol.set_selected(0);
    protocol.set_tooltip_text(Some(
        "Fast-35K: the calibrated cryo-UV protocol fixes the 50 K \u{2192} 310 K thermal cycle and \
         the step budget (the thermal shock drives cryptic-pocket opening); --temperature/ \
         --cryo-temp are not emitted. Standard: the run drops --fast + --multi-differential so the \
         engine's user-configurable protocol honors --temperature/--cryo-temp/--steps 1:1 \
         (nhs_rt_full.rs:5094-5112 / :9080).",
    ));
    proto_row.append(&proto_lbl);
    proto_row.append(&protocol);
    thermo.append(&proto_row);

    let temperature = labeled_spin(
        &thermo,
        "Warm / end-of-ramp target K (--temperature, Standard only)",
        4.0,
        1000.0,
        1.0,
        300.0,
    );
    temperature.set_sensitive(false); // live only in Standard mode
    temperature.set_tooltip_text(Some(
        "--temperature: the warm (end-of-ramp) target the STANDARD protocol ramps to \
         (nhs_rt_full.rs:404 default 300 K; consumed as end_temp at :5097 / :9055). Live only in \
         Standard mode \u{2014} under the calibrated Fast protocol the 50 K \u{2192} 310 K cycle is \
         fixed and this value is not emitted.",
    ));
    let cryo_temp = labeled_spin(
        &thermo,
        "Cryo start K (--cryo-temp, Standard only)",
        0.0,
        310.0,
        1.0,
        50.0,
    );
    cryo_temp.set_sensitive(false); // live only in Standard mode
    cryo_temp.set_tooltip_text(Some(
        "--cryo-temp: the cold start temperature the STANDARD protocol begins at \
         (nhs_rt_full.rs:408 default 50 K; consumed as start_temp at :5096 / :9054). Live only in \
         Standard mode \u{2014} under Fast it is fixed by calibration and not emitted.",
    ));
    let thermo_note = gtk::Label::new(Some(
        "Fast-35K (default): the 50 K \u{2192} 310 K thermal cycle + step budget are \
         calibration-fixed, so the setpoints above are greyed and not emitted. Standard: the \
         engine honors --temperature/--cryo-temp/--steps 1:1 (the base drops --fast + \
         --multi-differential so the user-configurable else-branch runs).",
    ));
    thermo_note.add_css_class("dim");
    thermo_note.set_wrap(true);
    thermo_note.set_xalign(0.0);
    thermo.append(&thermo_note);

    let steps = labeled_spin(
        &thermo,
        "Total MD steps (--steps)",
        1000.0,
        5_000_000.0,
        1000.0,
        500000.0,
    );
    steps.set_tooltip_text(Some(
        "--steps total MD steps (nhs_rt_full.rs:399, default 500000). Standard: honored directly \
         (steps_per_stream = args.steps, :9080). Fast: --steps alone is a no-op, so a non-default \
         value is emitted with --wall-bound-hysteresis (the 5-stage protocol then allocates it); \
         the default value emits nothing, so the default Fast run is byte-identical to before.",
    ));
    // --steps is ALWAYS live now (no opt-in toggle); the Protocol handler only
    // toggles the temperature/cryo sensitivity.
    notebook.append_page(&thermo, Some(&gtk::Label::new(Some("Thermodynamics"))));

    // Page 2 — Capture: --save-trajectory-interval.
    let capture = gtk::Box::new(gtk::Orientation::Vertical, 4);
    capture.set_margin_top(6);
    capture.set_margin_bottom(6);
    capture.set_margin_start(6);
    capture.set_margin_end(6);
    let save_trajectory_interval = labeled_spin(
        &capture,
        "Save trajectory interval (--save-trajectory-interval)",
        0.0,
        100000.0,
        100.0,
        0.0,
    );
    let cap_note = gtk::Label::new(Some(
        "0 = off; set >0 to write frames every N steps for explicit-solvent / MSM / visualization",
    ));
    cap_note.add_css_class("dim");
    cap_note.set_wrap(true);
    cap_note.set_xalign(0.0);
    capture.append(&cap_note);
    notebook.append_page(&capture, Some(&gtk::Label::new(Some("Capture"))));

    // Page 3 — Detection: --lining-cutoff (genuinely honored). --cluster-threshold is
    // intentionally NOT offered: the engine never consumes it (nhs_rt_full.rs:434 decl +
    // :3051 echo only) and --md-only-evidence replaces in-engine CCL clustering entirely.
    let detection = gtk::Box::new(gtk::Orientation::Vertical, 4);
    detection.set_margin_top(6);
    detection.set_margin_bottom(6);
    detection.set_margin_start(6);
    detection.set_margin_end(6);
    let lining_cutoff = labeled_spin(
        &detection,
        "Lining residue cutoff \u{00c5} (--lining-cutoff)",
        3.0,
        20.0,
        0.5,
        8.0,
    );
    let det_note = gtk::Label::new(Some(
        "Cluster threshold (--cluster-threshold) is intentionally not offered: the engine never \
         consumes it, and --md-only-evidence replaces in-engine CCL clustering with phase-manifold \
         ranking.",
    ));
    det_note.add_css_class("dim");
    det_note.set_wrap(true);
    det_note.set_xalign(0.0);
    detection.append(&det_note);
    notebook.append_page(&detection, Some(&gtk::Label::new(Some("Detection"))));

    (
        notebook,
        ParamWidgets {
            protocol,
            temperature,
            cryo_temp,
            steps,
            save_trajectory_interval,
            lining_cutoff,
        },
    )
}

/// Build the Run Engine panel.
pub fn build(ctx: &AppContext) -> gtk::Widget {
    let primary_manifest = std::env::var_os("PRISMOS_PRIMARY_WORKFLOW_MANIFEST").map(PathBuf::from);
    let primary_mode = primary_manifest.is_some();
    let root = gtk::Box::new(gtk::Orientation::Vertical, 12);
    root.add_css_class("prismos-root");
    root.set_margin_top(16);
    root.set_margin_bottom(16);
    root.set_margin_start(16);
    root.set_margin_end(16);

    let title = gtk::Label::new(Some("Run Engine — Full governed nhs_rt_full runtime"));
    title.add_css_class("prismos-title");
    title.set_halign(gtk::Align::Start);
    root.append(&title);

    let subtitle = gtk::Label::new(Some(
        "Pick a topology and output directory, then choose a calibrated typed preset or Complete \
         engine argv for the full installed nhs_rt_full surface. Launch uses the governed \
         preflight → engine → postflight adapter without an engine-option allowlist. On a passing \
         run the app Run Root is pointed at the output.",
    ));
    subtitle.add_css_class("subtitle");
    subtitle.set_halign(gtk::Align::Start);
    subtitle.set_wrap(true);
    root.append(&subtitle);

    // Primary mode accepts an explicitly selected production-prepared experiment.
    // Generic topology/output controls are not reinterpreted as a benchmark panel.
    let primary_card = gtk::Box::new(gtk::Orientation::Vertical, 8);
    primary_card.add_css_class("card");
    let primary_choose = gtk::Button::with_label("Choose prepared 1010 experiment…");
    let primary_panel_label = gtk::Label::new(Some("Select a fresh production-prepared panel.json directory. No native work runs during inspection."));
    primary_panel_label.set_wrap(true);
    primary_panel_label.set_xalign(0.0);
    primary_card.append(&primary_choose);
    primary_card.append(&primary_panel_label);
    primary_card.set_visible(primary_mode);
    root.append(&primary_card);
    let primary_results = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    root.append(&primary_results);
    if primary_mode {
        title.set_text("Run Engine — complete 1010 production workflow");
        subtitle.set_text("Choose a validated prepared 1010 experiment. Production owns capture, same-budget continuation, full PathA and PathB, fixed postseal scoring and verified archives. Generic target preparation remains separate; no topology is silently substituted.");
    }

    // ---- run-mode card (single debug run vs N-replicate consensus) ------
    // CLAUDE.md: "Single runs are for debugging. Production = replicate consensus."
    // Consensus mode drives scripts/prism_replicate.py (the canonical production
    // entry point); Single mode drives the single-run wrapper exactly as before.
    let mode_card = gtk::Box::new(gtk::Orientation::Vertical, 8);
    mode_card.add_css_class("card");
    let mode_heading = gtk::Label::new(Some("Run mode"));
    mode_heading.add_css_class("subtitle");
    mode_heading.set_halign(gtk::Align::Start);
    mode_card.append(&mode_heading);
    let mode_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let run_mode = gtk::DropDown::from_strings(&RUN_MODE_LABELS);
    run_mode.set_selected(0);
    let nrep_lbl = gtk::Label::new(Some("Replicates (--n-replicates)"));
    nrep_lbl.set_xalign(0.0);
    let n_replicates_spin = gtk::SpinButton::with_range(3.0, 9.0, 1.0);
    n_replicates_spin.set_value(5.0);
    // The replicate count is only meaningful in consensus mode.
    nrep_lbl.set_visible(false);
    n_replicates_spin.set_visible(false);
    mode_row.append(&run_mode);
    mode_row.append(&nrep_lbl);
    mode_row.append(&n_replicates_spin);
    mode_card.append(&mode_row);
    let mode_note = gtk::Label::new(Some(
        "Single run (debug): one engine pass through scripts/prism-validate-and-run.sh with the \
         surface / physics controls below. N-replicate consensus (production): drives \
         scripts/prism_replicate.py \u{2014} N replicates + cross-run metastable-pocket consensus \
         (CLAUDE.md canonical production entry point). Replicate orchestration retains its \
         calibrated profile; Complete engine argv remains the fully tunable single-run path.",
    ));
    mode_note.add_css_class("dim");
    mode_note.set_wrap(true);
    mode_note.set_xalign(0.0);
    mode_card.append(&mode_note);
    root.append(&mode_card);
    mode_card.set_visible(!primary_mode);

    // ---- selection card -------------------------------------------------
    let sel_card = gtk::Box::new(gtk::Orientation::Vertical, 8);
    sel_card.add_css_class("card");

    let topo_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let topo_btn = gtk::Button::with_label("Choose topology\u{2026}");
    let pdb_btn = gtk::Button::with_label("Load raw PDB\u{2026}");
    let topo_label = gtk::Label::new(Some(
        "no topology selected  (choose a *.topology.json, or Load raw PDB to run prep first)",
    ));
    topo_label.add_css_class("subtitle");
    topo_label.set_halign(gtk::Align::Start);
    topo_label.set_hexpand(true);
    topo_label.set_wrap(true);
    topo_label.set_xalign(0.0);
    topo_row.append(&topo_btn);
    topo_row.append(&pdb_btn);
    topo_row.append(&topo_label);
    sel_card.append(&topo_row);

    let out_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let out_btn = gtk::Button::with_label("Choose output\u{2026}");
    let output_label = gtk::Label::new(Some("no output directory selected"));
    output_label.add_css_class("subtitle");
    output_label.set_halign(gtk::Align::Start);
    output_label.set_hexpand(true);
    output_label.set_wrap(true);
    output_label.set_xalign(0.0);
    out_row.append(&out_btn);
    out_row.append(&output_label);
    sel_card.append(&out_row);

    // multi-stream sizing toggles
    let stream_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let stream_lbl = gtk::Label::new(Some("Multi-stream"));
    stream_lbl.set_width_chars(14);
    stream_lbl.set_xalign(0.0);
    let stream8 = gtk::ToggleButton::with_label("8  (\u{2264}400 res)");
    let stream20 = gtk::ToggleButton::with_label("20  (>400 res)");
    stream20.set_group(Some(&stream8));
    stream8.set_active(true);
    let residue_note = gtk::Label::new(Some("residue count unknown until a topology is chosen"));
    residue_note.add_css_class("dim");
    residue_note.set_halign(gtk::Align::Start);
    residue_note.set_hexpand(true);
    residue_note.set_xalign(0.0);
    stream_row.append(&stream_lbl);
    stream_row.append(&stream8);
    stream_row.append(&stream20);
    stream_row.append(&residue_note);
    sel_card.append(&stream_row);

    root.append(&sel_card);
    sel_card.set_visible(!primary_mode);

    // ---- PDB intake + staged prep card ---------------------------------
    let prep_card = gtk::Box::new(gtk::Orientation::Vertical, 8);
    prep_card.add_css_class("card");

    let prep_heading = gtk::Label::new(Some("Prep pipeline (raw PDB \u{2192} topology)"));
    prep_heading.add_css_class("subtitle");
    prep_heading.set_halign(gtk::Align::Start);
    prep_card.append(&prep_heading);

    let pdb_label = gtk::Label::new(Some(
        "No raw PDB loaded. Loading a *.pdb runs the real staged prep (clean \u{2192} [merge] \u{2192} \
         prep) before a governed engine run. Each phase is fired by YOU; nothing auto-runs.",
    ));
    pdb_label.add_css_class("dim");
    pdb_label.set_halign(gtk::Align::Start);
    pdb_label.set_wrap(true);
    pdb_label.set_xalign(0.0);
    prep_card.append(&pdb_label);

    let chain_label = gtk::Label::new(Some("chains: (load a PDB to scan)"));
    chain_label.add_css_class("dim");
    chain_label.set_halign(gtk::Align::Start);
    chain_label.set_wrap(true);
    chain_label.set_xalign(0.0);
    prep_card.append(&chain_label);

    // Multichain decision radio (CLAUDE.md Multichain procedure).
    let decision_row = gtk::Box::new(gtk::Orientation::Vertical, 2);
    let dec_a = gtk::CheckButton::with_label(
        "(a) interface / cryptic site at a chain interface \u{2192} merge chains",
    );
    let dec_b = gtk::CheckButton::with_label("(b) chains independent \u{2192} one chain per run");
    dec_b.set_group(Some(&dec_a));
    let dec_c = gtk::CheckButton::with_label(
        "(c) one chain is ligand/peptide/cofactor \u{2192} keep protein chain only",
    );
    dec_c.set_group(Some(&dec_a));
    decision_row.append(&dec_a);
    decision_row.append(&dec_b);
    decision_row.append(&dec_c);
    decision_row.set_visible(false);
    prep_card.append(&decision_row);

    // Phase B: Clean
    let clean_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let clean_btn = gtk::Button::with_label("Run Clean");
    clean_btn.set_sensitive(false);
    let clean_chip = make_chip("Clean: idle");
    clean_row.append(&clean_btn);
    clean_row.append(&clean_chip);
    let clean_note = gtk::Label::new(Some(
        "prism-clean.py: strips altconfs + ALL HETATM (\u{2192} apo), residue-type diversity \
         \u{2265}15 hard gate.",
    ));
    clean_note.add_css_class("dim");
    clean_note.set_hexpand(true);
    clean_note.set_wrap(true);
    clean_note.set_xalign(0.0);
    clean_row.append(&clean_note);
    prep_card.append(&clean_row);

    // Phase B1/B2: Merge (multichain Merge decision only)
    let merge_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let merge_btn = gtk::Button::with_label("Run Merge");
    merge_btn.set_sensitive(false);
    let merge_chip = make_chip("Merge: n/a");
    merge_row.append(&merge_btn);
    merge_row.append(&merge_chip);
    let merge_note = gtk::Label::new(Some(
        "prism-merge-chains.py: renumbers all residues to chain A seq from 1; chain_map.json is \
         the ONLY back-translation (forwarded as --chain-map).",
    ));
    merge_note.add_css_class("dim");
    merge_note.set_hexpand(true);
    merge_note.set_wrap(true);
    merge_note.set_xalign(0.0);
    merge_row.append(&merge_note);
    prep_card.append(&merge_row);

    // Phase C: Prep topology
    let prep_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let prep_btn = gtk::Button::with_label("Run Prep topology");
    prep_btn.set_sensitive(false);
    let prep_chip = make_chip("Prep: idle");
    prep_row.append(&prep_btn);
    prep_row.append(&prep_chip);
    let prep_note = gtk::Label::new(Some(
        "prism-prep <in.pdb> <out.topology.json> --mode cryptic. Stage-2 verify is mandatory; \
         PASS iff the topology.json exists afterwards.",
    ));
    prep_note.add_css_class("dim");
    prep_note.set_hexpand(true);
    prep_note.set_wrap(true);
    prep_note.set_xalign(0.0);
    prep_row.append(&prep_note);
    prep_card.append(&prep_row);

    let advisory_label = gtk::Label::new(Some(""));
    advisory_label.add_css_class("dim");
    advisory_label.set_halign(gtk::Align::Start);
    advisory_label.set_wrap(true);
    advisory_label.set_xalign(0.0);
    advisory_label.set_visible(false);
    prep_card.append(&advisory_label);

    prep_card.set_visible(false); // shown only once a raw PDB is loaded
    root.append(&prep_card);
    prep_card.set_visible(!primary_mode);

    // ---- Surface & Mode selector card ----------------------------------
    let (surface_card, surface_widgets) = build_surface_card();
    root.append(&surface_card);
    surface_card.set_visible(!primary_mode);
    let complete_argv_widgets = build_complete_argv_card();
    root.append(&complete_argv_widgets.card);
    complete_argv_widgets.card.set_visible(!primary_mode);

    // ---- Physics parameter tabs (Thermodynamics / Capture / Detection) --
    // FUNDAMENTAL engine knobs (temperature / step budget / clustering geometry)
    // that apply to BOTH the Canonical (ext) and named-engine-surface run paths —
    // a named surface never overrides them, so these tabs are never greyed.
    let params_heading = gtk::Label::new(Some("Physics parameters"));
    params_heading.add_css_class("subtitle");
    params_heading.set_halign(gtk::Align::Start);
    params_heading.set_margin_top(4);
    root.append(&params_heading);
    params_heading.set_visible(!primary_mode);
    let (params_notebook, param_widgets) = build_params_tabs();
    root.append(&params_notebook);
    params_notebook.set_visible(!primary_mode);

    // ---- command preview ------------------------------------------------
    let cmd_heading = gtk::Label::new(Some("Exact governed command"));
    cmd_heading.add_css_class("subtitle");
    cmd_heading.set_halign(gtk::Align::Start);
    cmd_heading.set_margin_top(4);
    root.append(&cmd_heading);

    let cmd_preview = gtk::Label::new(Some(
        "Choose a topology and output directory to see the exact governed run command.",
    ));
    cmd_preview.add_css_class("mono");
    cmd_preview.add_css_class("argv");
    cmd_preview.set_halign(gtk::Align::Start);
    cmd_preview.set_xalign(0.0);
    cmd_preview.set_wrap(false);
    cmd_preview.set_selectable(true);
    let cmd_scroll = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Never)
        .child(&cmd_preview)
        .build();
    cmd_scroll.add_css_class("run-command-preview");
    root.append(&cmd_scroll);

    // ---- action row + gate chips ---------------------------------------
    let action_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let launch_btn = gtk::Button::with_label("Launch");
    launch_btn.add_css_class("accent-button");
    launch_btn.set_sensitive(false);
    let cancel_btn = gtk::Button::with_label("Cancel");
    cancel_btn.set_sensitive(false);
    action_row.append(&launch_btn);
    action_row.append(&cancel_btn);

    let gate_strip = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    gate_strip.add_css_class("gate-strip");
    gate_strip.set_hexpand(true);
    gate_strip.set_halign(gtk::Align::End);
    let preflight_chip = make_chip("Preflight: idle");
    let engine_chip = make_chip("Engine: idle");
    let postflight_chip = make_chip("Postflight: idle");
    gate_strip.append(&preflight_chip);
    gate_strip.append(&engine_chip);
    gate_strip.append(&postflight_chip);
    action_row.append(&gate_strip);
    root.append(&action_row);

    // ---- live log -------------------------------------------------------
    let log_view = gtk::TextView::new();
    log_view.set_editable(false);
    log_view.set_cursor_visible(false);
    log_view.set_monospace(true);
    log_view.set_wrap_mode(gtk::WrapMode::WordChar);
    log_view.add_css_class("mono");
    log_view.add_css_class("log-view");
    log_view
        .buffer()
        .set_text("Idle. Choose a topology + output, then click Launch to start a governed run.\n");
    let log_scroll = gtk::ScrolledWindow::builder()
        .min_content_height(300)
        .vexpand(true)
        .hexpand(true)
        .child(&log_view)
        .build();
    root.append(&log_scroll);

    // ---- resolver precheck: governed adapter + installed engine ----------
    let script_path = ctx.resolver.resolve(PrismBin::PrismValidateRun);
    let engine_path = ctx.resolver.resolve(PrismBin::NhsRtFull);
    let script_ok = script_path.is_file() && engine_path.is_file();

    let ui = Rc::new(RunEngine {
        ctx: ctx.clone(),
        primary_manifest,
        primary_panel: RefCell::new(None),
        primary_panel_verified: Cell::new(false),
        primary_panel_label: primary_panel_label.clone(),
        primary_results: primary_results.clone(),
        topo: RefCell::new(None),
        output: RefCell::new(None),
        multi_stream: Cell::new(8),
        consensus_mode: Cell::new(false),
        n_replicates: Cell::new(5),
        raw_pdb: RefCell::new(None),
        chains: RefCell::new(Vec::new()),
        cleaned_pdb: RefCell::new(None),
        merged_pdb: RefCell::new(None),
        chain_map: RefCell::new(None),
        prep_phase: Cell::new(PrepPhase::Idle),
        decision: Cell::new(None),
        prep_clean_ok: Cell::new(false),
        prep_prep_ok: Cell::new(false),
        prep_merge_ok: Cell::new(false),
        preset: Cell::new(Preset::Canonical),
        engine_surface: Cell::new(EngineSurface::None),
        max_wall: Cell::new(180),
        spike_percentile: Cell::new(70),
        rest2: Cell::new(false),
        multi_temp: Cell::new(false),
        stepped_holds: Cell::new(false),
        clustering_backend: RefCell::new(None),
        ghost_so3: Cell::new(0.75),
        ghost_cell_a: Cell::new(5.0),
        ghost_max_temporal: Cell::new(500),
        ghost_step_bucket: Cell::new(500),
        standard_protocol: Cell::new(false),
        temperature: Cell::new(300.0),
        cryo_temp: Cell::new(50.0),
        steps: Cell::new(500_000),
        save_traj_interval: Cell::new(0),
        lining_cutoff: Cell::new(8.0),
        engine_argv: RefCell::new(EngineArgv::default()),
        engine_argv_error: RefCell::new(None),
        active_md_only_required: Cell::new(true),
        active_consensus_required: Cell::new(false),
        running: Cell::new(false),
        cancelled: Cell::new(false),
        job: RefCell::new(None),
        postflight_pass: Cell::new(None),
        script_ok: Cell::new(script_ok),
        topo_label: topo_label.clone(),
        output_label: output_label.clone(),
        residue_note: residue_note.clone(),
        cmd_preview: cmd_preview.clone(),
        preflight_chip: preflight_chip.clone(),
        engine_chip: engine_chip.clone(),
        postflight_chip: postflight_chip.clone(),
        launch_btn: launch_btn.clone(),
        cancel_btn: cancel_btn.clone(),
        stream8: stream8.clone(),
        stream20: stream20.clone(),
        log_view: log_view.clone(),
        pdb_label: pdb_label.clone(),
        chain_label: chain_label.clone(),
        decision_row: decision_row.clone(),
        clean_chip: clean_chip.clone(),
        merge_chip: merge_chip.clone(),
        prep_chip: prep_chip.clone(),
        clean_btn: clean_btn.clone(),
        merge_btn: merge_btn.clone(),
        prep_btn: prep_btn.clone(),
        advisory_label: advisory_label.clone(),
    });

    // Do not let an ordinary window close orphan a leased production controller.
    // An uncontrolled OS/process kill remains the existing OPEN-budget recovery case.
    {
        let ui_close = ui.clone();
        ui.ctx.window.connect_close_request(move |_| {
            if ui_close.primary_manifest.is_some() && ui_close.running.get() {
                cancel(&ui_close);
                ui_close.ctx.toast("Waiting for production cleanup; close after the controller exits");
                glib::Propagation::Stop
            } else { glib::Propagation::Proceed }
        });
    }

    if !script_path.is_file() {
        append_log(
            &ui.log_view,
            &format!(
                "\u{26A0}  Entry-point script not found at {} — Launch is disabled. \
                 Set PRISMOS_UI_REPO_ROOT or PRISMOS_UI_BIN_PRISM_VALIDATE_AND_RUN_SH to the \
                 repo whose scripts/ holds prism-validate-and-run.sh.",
                script_path.display()
            ),
        );
    }
    if !engine_path.is_file() {
        append_log(
            &ui.log_view,
            &format!(
                "\u{26A0}  Installed engine not found at {} — Launch is disabled. \
                 PRISM_ENGINE_BIN must resolve to /opt/prismos/bin/nhs_rt_full on the appliance.",
                engine_path.display()
            ),
        );
    } else {
        append_log(
            &ui.log_view,
            &format!(
                "\u{2713}  installed engine binding: {}",
                engine_path.display()
            ),
        );
    }

    // Explicit prepared-panel selection is independent of the old topology/output chooser.
    {
        let ui = ui.clone();
        primary_choose.connect_clicked(move |_| {
            let dialog = gtk::FileDialog::new();
            dialog.set_title("Choose a production-prepared 1010 experiment directory");
            let ui = ui.clone();
            let window = ui.ctx.window.clone();
            dialog.select_folder(Some(&window), gtk::gio::Cancellable::NONE, move |result| {
                if let Ok(file) = result {
                    if let Some(path) = file.path() { inspect_primary_panel(&ui, path); }
                }
            });
        });
    }

    // topology chooser
    {
        let ui = ui.clone();
        topo_btn.connect_clicked(move |_| {
            let dialog = gtk::FileDialog::new();
            dialog.set_title("Choose a topology (*.topology.json)");
            let filter = gtk::FileFilter::new();
            filter.set_name(Some("Topology JSON (*.topology.json)"));
            filter.add_pattern("*.topology.json");
            filter.add_pattern("*.json");
            dialog.set_default_filter(Some(&filter));
            let ui = ui.clone();
            let window = ui.ctx.window.clone();
            dialog.open(Some(&window), gtk::gio::Cancellable::NONE, move |result| {
                if let Ok(file) = result {
                    if let Some(path) = file.path() {
                        // A directly-chosen topology SKIPS prep: clear any prior
                        // prep-derived chain map so it is not appended to this run.
                        *ui.chain_map.borrow_mut() = None;
                        ui.prep_phase.set(PrepPhase::Idle);
                        set_topology(&ui, path);
                    }
                }
            });
        });
    }

    // output chooser (folder)
    {
        let ui = ui.clone();
        out_btn.connect_clicked(move |_| {
            let dialog = gtk::FileDialog::new();
            dialog.set_title("Choose an output directory for this run");
            let ui = ui.clone();
            let window = ui.ctx.window.clone();
            dialog.select_folder(Some(&window), gtk::gio::Cancellable::NONE, move |result| {
                if let Ok(file) = result {
                    if let Some(path) = file.path() {
                        *ui.output.borrow_mut() = Some(path.clone());
                        ui.output_label.set_text(&path.display().to_string());
                        refresh_command(&ui);
                    }
                }
            });
        });
    }

    // multi-stream toggles
    {
        let ui = ui.clone();
        stream8.connect_toggled(move |b| {
            if b.is_active() {
                ui.multi_stream.set(8);
                refresh_command(&ui);
            }
        });
    }
    {
        let ui = ui.clone();
        stream20.connect_toggled(move |b| {
            if b.is_active() {
                ui.multi_stream.set(20);
                refresh_command(&ui);
            }
        });
    }

    // run-mode dropdown (single debug run vs N-replicate consensus) + replicate count
    {
        let ui = ui.clone();
        let nrep_lbl = nrep_lbl.clone();
        let n_replicates_spin = n_replicates_spin.clone();
        run_mode.connect_selected_notify(move |d| {
            let consensus = d.selected() == 1;
            ui.consensus_mode.set(consensus);
            // The replicate spin + its label are only shown in consensus mode.
            nrep_lbl.set_visible(consensus);
            n_replicates_spin.set_visible(consensus);
            refresh_command(&ui);
        });
    }
    {
        let ui = ui.clone();
        n_replicates_spin.connect_value_changed(move |s| {
            ui.n_replicates.set(s.value_as_int().max(1) as usize);
            refresh_command(&ui);
        });
    }

    // Complete engine argv editor: one exact token per non-empty line. Parsing
    // never invokes a shell; an invalid reserved transport token disables Launch.
    {
        let ui = ui.clone();
        let status = complete_argv_widgets.status.clone();
        complete_argv_widgets
            .editor
            .buffer()
            .connect_changed(move |buffer| {
                let text = buffer.text(&buffer.start_iter(), &buffer.end_iter(), true);
                match EngineArgv::from_token_lines(text.as_str()) {
                    Ok(argv) => {
                        status.remove_css_class("error");
                        status.set_text(&format!(
                            "{} additional engine argv tokens — exact order preserved",
                            argv.len()
                        ));
                        *ui.engine_argv.borrow_mut() = argv;
                        *ui.engine_argv_error.borrow_mut() = None;
                    }
                    Err(error) => {
                        status.add_css_class("error");
                        status.set_text(&format!("Engine argv rejected: {error}"));
                        *ui.engine_argv_error.borrow_mut() = Some(error);
                    }
                }
                refresh_command(&ui);
            });
    }

    // launch
    {
        let ui = ui.clone();
        launch_btn.connect_clicked(move |_| launch(&ui));
    }

    // cancel
    {
        let ui = ui.clone();
        cancel_btn.connect_clicked(move |_| cancel(&ui));
    }

    // ---- PDB intake -----------------------------------------------------
    {
        let ui = ui.clone();
        let prep_card = prep_card.clone();
        let dec_a = dec_a.clone();
        let dec_b = dec_b.clone();
        let dec_c = dec_c.clone();
        pdb_btn.connect_clicked(move |_| {
            let dialog = gtk::FileDialog::new();
            dialog.set_title("Load a raw PDB (*.pdb) to run prep");
            let filter = gtk::FileFilter::new();
            filter.set_name(Some("Protein Data Bank (*.pdb)"));
            filter.add_pattern("*.pdb");
            dialog.set_default_filter(Some(&filter));
            let ui = ui.clone();
            let prep_card = prep_card.clone();
            let dec_a = dec_a.clone();
            let dec_b = dec_b.clone();
            let dec_c = dec_c.clone();
            let window = ui.ctx.window.clone();
            dialog.open(Some(&window), gtk::gio::Cancellable::NONE, move |result| {
                if let Ok(file) = result {
                    if let Some(path) = file.path() {
                        set_raw_pdb(&ui, path, &prep_card, &dec_a, &dec_b, &dec_c);
                    }
                }
            });
        });
    }

    // multichain decision radios
    for (btn, dec) in [
        (&dec_a, MultichainDecision::Merge),
        (&dec_b, MultichainDecision::Independent),
        (&dec_c, MultichainDecision::LigandProtein),
    ] {
        let ui = ui.clone();
        btn.connect_toggled(move |b| {
            if b.is_active() {
                ui.decision.set(Some(dec));
                refresh_prep_sensitivity(&ui);
            }
        });
    }

    // prep phase buttons
    {
        let ui = ui.clone();
        clean_btn.connect_clicked(move |_| run_clean_phase(&ui));
    }
    {
        let ui = ui.clone();
        merge_btn.connect_clicked(move |_| run_merge_phase(&ui));
    }
    {
        let ui = ui.clone();
        prep_btn.connect_clicked(move |_| run_prep_phase(&ui));
    }

    // ---- surface widget handlers ---------------------------------------
    {
        let ui = ui.clone();
        let params_notebook = params_notebook.clone();
        let typed_controls = surface_widgets.typed_controls.clone();
        let complete_argv_card = complete_argv_widgets.card.clone();
        surface_widgets.preset.connect_selected_notify(move |d| {
            let preset = preset_from_index(d.selected());
            ui.preset.set(preset);
            let complete = preset == Preset::CompleteArgv;
            typed_controls.set_sensitive(!complete);
            complete_argv_card.set_visible(complete);
            // The run-parameter tabs apply only to the Canonical (or any named
            // engine-surface) run path; twin/smoke presets ignore them, so grey
            // the tabs out under those presets to avoid a dead-click.
            params_notebook.set_sensitive(
                !complete
                    && (preset == Preset::Canonical
                        || ui.engine_surface.get() != EngineSurface::None),
            );
            refresh_command(&ui);
        });
    }
    {
        let ui = ui.clone();
        let rest2 = surface_widgets.rest2.clone();
        let multi_temp = surface_widgets.multi_temp.clone();
        let stepped_holds = surface_widgets.stepped_holds.clone();
        let spike_percentile = surface_widgets.spike_percentile.clone();
        let clustering = surface_widgets.clustering.clone();
        let ghost_box = surface_widgets.ghost_box.clone();
        let preset = surface_widgets.preset.clone();
        let params_notebook = params_notebook.clone();
        surface_widgets
            .engine_surface
            .connect_selected_notify(move |d| {
                let surface = engine_surface_from_index(d.selected());
                ui.engine_surface.set(surface);
                // A named engine surface EXPANDS upstream of every guard/dispatch
                // and OVERRIDES the command-class preset AND the advanced-surface
                // tail, so grey the 9 widgets it supersedes (rest2/multi_temp/
                // stepped_holds/spike_percentile/clustering + the 4 ghost spins inside
                // ghost_box) AND the preset dropdown. The physics parameter tabs are
                // NOT greyed — they are fundamental and apply under a named surface too.
                let overridden = surface != EngineSurface::None;
                rest2.set_sensitive(!overridden);
                multi_temp.set_sensitive(!overridden);
                stepped_holds.set_sensitive(!overridden);
                spike_percentile.set_sensitive(!overridden);
                clustering.set_sensitive(!overridden);
                ghost_box.set_sensitive(
                    !overridden
                        && ui.clustering_backend.borrow().as_deref()
                            == Some("ghost-phase-lattice-4d"),
                );
                preset.set_sensitive(!overridden);
                // Params apply under a named surface OR the Canonical preset.
                params_notebook.set_sensitive(overridden || ui.preset.get() == Preset::Canonical);
                refresh_command(&ui);
            });
    }
    {
        let ui = ui.clone();
        let ghost_box = surface_widgets.ghost_box.clone();
        surface_widgets
            .clustering
            .connect_selected_notify(move |d| {
                let label = CLUSTERING_LABELS
                    .get(d.selected() as usize)
                    .copied()
                    .unwrap_or("auto");
                if label == "auto" {
                    *ui.clustering_backend.borrow_mut() = None;
                } else {
                    *ui.clustering_backend.borrow_mut() = Some(label.to_string());
                }
                // Only enable ghost_box when the ghost backend is chosen AND no named
                // surface is overriding the advanced tail (so the two ghost_box writers
                // can never desync).
                ghost_box.set_sensitive(
                    label == "ghost-phase-lattice-4d"
                        && ui.engine_surface.get() == EngineSurface::None,
                );
                refresh_command(&ui);
            });
    }
    {
        let ui = ui.clone();
        surface_widgets.rest2.connect_toggled(move |b| {
            ui.rest2.set(b.is_active());
            refresh_command(&ui);
        });
    }
    {
        let ui = ui.clone();
        surface_widgets.multi_temp.connect_toggled(move |b| {
            ui.multi_temp.set(b.is_active());
            refresh_command(&ui);
        });
    }
    {
        let ui = ui.clone();
        surface_widgets.stepped_holds.connect_toggled(move |b| {
            ui.stepped_holds.set(b.is_active());
            refresh_command(&ui);
        });
    }
    {
        let ui = ui.clone();
        surface_widgets.max_wall.connect_value_changed(move |s| {
            ui.max_wall.set(s.value_as_int().max(1) as u64);
            refresh_command(&ui);
        });
    }
    {
        let ui = ui.clone();
        surface_widgets
            .spike_percentile
            .connect_value_changed(move |s| {
                ui.spike_percentile.set(s.value_as_int().max(1) as u32);
                refresh_command(&ui);
            });
    }
    {
        let ui = ui.clone();
        surface_widgets.ghost_so3.connect_value_changed(move |s| {
            ui.ghost_so3.set(s.value() as f32);
            refresh_command(&ui);
        });
    }
    {
        let ui = ui.clone();
        surface_widgets
            .ghost_cell_a
            .connect_value_changed(move |s| {
                ui.ghost_cell_a.set(s.value() as f32);
                refresh_command(&ui);
            });
    }
    {
        let ui = ui.clone();
        surface_widgets
            .ghost_max_temporal
            .connect_value_changed(move |s| {
                ui.ghost_max_temporal.set(s.value_as_int().max(1) as u64);
                refresh_command(&ui);
            });
    }
    {
        let ui = ui.clone();
        surface_widgets
            .ghost_step_bucket
            .connect_value_changed(move |s| {
                ui.ghost_step_bucket.set(s.value_as_int().max(1) as u64);
                refresh_command(&ui);
            });
    }

    // ---- run-parameter tab handlers (protocol-aware) -------------------
    // The Protocol dropdown selects Fast (calibrated) vs Standard (user-configurable
    // thermodynamics). On Standard the engine honors --temperature/--cryo-temp/--steps
    // 1:1 (nhs_rt_full.rs:5094-5112 / :9080), so those spins go live; on Fast the
    // thermal cycle is calibration-fixed, so temperature/cryo are greyed + not emitted.
    // --steps is ALWAYS live (no opt-in toggle). --cluster-threshold is not offered.
    {
        let ui = ui.clone();
        let temperature = param_widgets.temperature.clone();
        let cryo_temp = param_widgets.cryo_temp.clone();
        let steps = param_widgets.steps.clone();
        param_widgets.protocol.connect_selected_notify(move |d| {
            let standard = d.selected() == 1;
            ui.standard_protocol.set(standard);
            // Standard un-greys temperature/cryo; Fast greys them back. --steps stays
            // always live in both modes.
            temperature.set_sensitive(standard);
            cryo_temp.set_sensitive(standard);
            steps.set_sensitive(true);
            refresh_command(&ui);
        });
    }
    {
        let ui = ui.clone();
        param_widgets.temperature.connect_value_changed(move |s| {
            ui.temperature.set(s.value() as f32);
            refresh_command(&ui);
        });
    }
    {
        let ui = ui.clone();
        param_widgets.cryo_temp.connect_value_changed(move |s| {
            ui.cryo_temp.set(s.value() as f32);
            refresh_command(&ui);
        });
    }
    {
        let ui = ui.clone();
        param_widgets.steps.connect_value_changed(move |s| {
            ui.steps.set(s.value_as_int() as i32);
            refresh_command(&ui);
        });
    }
    {
        let ui = ui.clone();
        param_widgets
            .save_trajectory_interval
            .connect_value_changed(move |s| {
                ui.save_traj_interval.set(s.value_as_int().max(0) as u32);
                refresh_command(&ui);
            });
    }
    {
        let ui = ui.clone();
        param_widgets.lining_cutoff.connect_value_changed(move |s| {
            ui.lining_cutoff.set(s.value() as f32);
            refresh_command(&ui);
        });
    }

    update_launch_sensitivity(&ui);

    let scroller = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vexpand(true)
        .hexpand(true)
        .child(&root)
        .build();
    scroller.upcast()
}

/// Record a chosen topology, derive the default output dir + multi-stream from
/// its residue count, and refresh the command preview.
fn set_topology(ui: &Rc<RunEngine>, path: PathBuf) {
    ui.topo_label.set_text(&path.display().to_string());
    *ui.topo.borrow_mut() = Some(path.clone());

    match crate::adapters::governed_output_for_topology(&path) {
        Ok(output) => {
            ui.output_label.set_text(&output.display().to_string());
            *ui.output.borrow_mut() = Some(output);
        }
        Err(error) => {
            *ui.output.borrow_mut() = None;
            ui.output_label.set_text(&format!("OUTPUT_POLICY_UNAVAILABLE: {error}"));
        }
    }

    // Residue-count-driven multi-stream sizing (honest degradation if unknown).
    match read_n_residues(&path) {
        Some(n) if n > MULTI_STREAM_LARGE_THRESHOLD => {
            ui.stream20.set_active(true);
            ui.multi_stream.set(20);
            ui.residue_note.set_text(&format!(
                "{n} residues \u{2192} --multi-stream 20 (>400 rule)"
            ));
        }
        Some(n) => {
            ui.stream8.set_active(true);
            ui.multi_stream.set(8);
            ui.residue_note.set_text(&format!(
                "{n} residues \u{2192} --multi-stream 8 (\u{2264}400 rule)"
            ));
        }
        None => {
            ui.residue_note.set_text(
                "residue count unknown (no n_residues in topology) \u{2014} defaulting to 8; \
                 override to 20 for a >400-residue target",
            );
        }
    }
    refresh_command(ui);
}

/// The engine's file-stem prefix: strip a trailing `.topology.json`.
fn topology_prefix(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "run".to_string());
    name.strip_suffix(".topology.json")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            path.file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "run".to_string())
        })
}

/// Read `n_residues` from a topology JSON, if present.
fn read_n_residues(path: &Path) -> Option<i64> {
    let text = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("n_residues").and_then(|x| x.as_i64())
}

// ========================================================================
// PDB intake + staged, gated prep pipeline.
//
// Each phase is one operator-fired, gated job (`spawn_cancellable`), run
// sequentially; the app NEVER auto-advances between phases and NEVER fires a
// prep or the engine as a side effect. On FINAL prep PASS the produced topology
// is funneled into `set_topology` — the SAME single launch path.
// ========================================================================

/// Read-only, pure-Rust chain scan: sorted-unique of column 22 (byte 21) over
/// `ATOM` records — the EXACT column `prism-clean.py:29` / `prism-merge-chains.py`
/// use. This is file I/O, NOT a script invocation, so it violates no script
/// execution policy.
fn scan_chains(path: &Path) -> Vec<String> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let mut seen = std::collections::BTreeSet::new();
    for line in text.lines() {
        if !line.starts_with("ATOM") {
            continue;
        }
        if let Some(&b) = line.as_bytes().get(21) {
            let c = (b as char).to_string();
            let c = c.trim().to_string();
            if !c.is_empty() {
                seen.insert(c);
            }
        }
    }
    seen.into_iter().collect()
}

/// The clean file-stem for a raw PDB: strip `.pdb`, then a trailing `_raw`.
fn pdb_stem(raw: &Path) -> String {
    let stem = raw
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "input".to_string());
    stem.strip_suffix("_raw")
        .map(|s| s.to_string())
        .unwrap_or(stem)
}

/// Directory to write prep intermediates into (next to the raw PDB).
fn prep_dir(raw: &Path) -> PathBuf {
    raw.parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Static, advisory-only CYS / residue-offset note keyed off the stem, drawn
/// from CLAUDE.md's CYS requirements registry + TOPOLOGY RESIDUE OFFSETS. Never
/// blocking (matches the wrapper's WARN-only preflight posture).
fn static_advisory(stem: &str) -> Option<String> {
    let s = stem.to_ascii_lowercase();
    let mut notes: Vec<&str> = Vec::new();
    if s.contains("1bzj") {
        notes.push("1bzj: CYS REQUIRED (PTP1B catalytic Cys215); topology = PDB - 1");
    }
    if s.contains("1p38") {
        notes.push("1p38: CYS REQUIRED (p38 MAPK)");
    }
    if s.contains("2hnp") {
        notes.push("2hnp: CYS REQUIRED (EphB2)");
    }
    if s.contains("1r3m") {
        notes.push("1r3m: CYS REQUIRED (RNase disulfides); obligate dimer — consider merge");
    }
    if s.contains("2iyt") {
        notes.push("2iyt: CYS NOT REQUIRED (Shikimate Kinase)");
    }
    if s.contains("1btl") {
        notes.push("1btl: topology residue offset -26 vs PDB author numbering");
    }
    if s.contains("4obe") {
        notes.push("4obe: KRAS G12C — verify CYS/GLY at topology position 12");
    }
    if notes.is_empty() {
        None
    } else {
        Some(format!("Advisory (non-blocking): {}", notes.join("  |  ")))
    }
}

/// Record a chosen raw PDB, scan its chains, reset the prep state machine, and
/// reveal the prep card. Clears any prior topology so Launch stays disabled
/// until prep produces a valid one.
fn set_raw_pdb(
    ui: &Rc<RunEngine>,
    path: PathBuf,
    prep_card: &gtk::Box,
    dec_a: &gtk::CheckButton,
    dec_b: &gtk::CheckButton,
    dec_c: &gtk::CheckButton,
) {
    let chains = scan_chains(&path);
    let stem = pdb_stem(&path);

    *ui.raw_pdb.borrow_mut() = Some(path.clone());
    *ui.chains.borrow_mut() = chains.clone();
    *ui.cleaned_pdb.borrow_mut() = None;
    *ui.merged_pdb.borrow_mut() = None;
    *ui.chain_map.borrow_mut() = None;
    ui.prep_clean_ok.set(false);
    ui.prep_merge_ok.set(false);
    ui.prep_prep_ok.set(false);
    ui.decision.set(None);
    ui.prep_phase.set(PrepPhase::NeedClean);

    // A raw PDB supersedes any directly-chosen topology.
    *ui.topo.borrow_mut() = None;
    ui.topo_label
        .set_text("topology will be produced by the prep pipeline below");

    prep_card.set_visible(true);
    ui.pdb_label
        .set_text(&format!("Raw PDB: {}", path.display()));
    set_chip(&ui.clean_chip, "Clean: idle", "info");
    set_chip(&ui.prep_chip, "Prep: idle", "info");

    let multichain = chains.len() > 1;
    if multichain {
        ui.chain_label.set_text(&format!(
            "{} chains: {}  \u{2014} choose the multichain decision below",
            chains.len(),
            chains.join(", ")
        ));
        ui.decision_row.set_visible(true);
        set_chip(&ui.merge_chip, "Merge: pending decision", "info");
        // Reset the radio to "no decision".
        dec_a.set_active(false);
        dec_b.set_active(false);
        dec_c.set_active(false);
    } else {
        let only = chains.first().cloned().unwrap_or_else(|| "A".to_string());
        ui.chain_label
            .set_text(&format!("single chain: {only}  \u{2014} single-chain prep"));
        ui.decision_row.set_visible(false);
        set_chip(&ui.merge_chip, "Merge: n/a (single chain)", "info");
    }

    match static_advisory(&stem) {
        Some(note) => {
            ui.advisory_label.set_text(&note);
            ui.advisory_label.set_visible(true);
        }
        None => ui.advisory_label.set_visible(false),
    }

    refresh_prep_sensitivity(ui);
    refresh_command(ui);
}

/// Whether the operator's current inputs put us on the multichain MERGE branch.
fn is_merge_branch(ui: &Rc<RunEngine>) -> bool {
    ui.chains.borrow().len() > 1 && ui.decision.get() == Some(MultichainDecision::Merge)
}

/// Recompute prep-phase button sensitivity: each button disables honestly when
/// its tool is missing, when a job is in flight, or when its gate is not yet met.
fn refresh_prep_sensitivity(ui: &Rc<RunEngine>) {
    let running = ui.running.get();
    let has_raw = ui.raw_pdb.borrow().is_some();
    let multichain = ui.chains.borrow().len() > 1;
    let decided = ui.decision.get().is_some();
    let merge_branch = is_merge_branch(ui);

    let clean_tool = ui.ctx.resolver.resolve(PrismBin::PrismClean).is_file();
    let merge_tool = ui
        .ctx
        .resolver
        .resolve(PrismBin::PrismMergeChains)
        .is_file();
    let prep_tool = ui.ctx.resolver.resolve(PrismBin::PrismPrep).is_file();

    // Clean: needs a raw PDB, the tool, and (for multichain) a decision.
    ui.clean_btn
        .set_sensitive(!running && has_raw && clean_tool && (!multichain || decided));
    // Merge: only on the merge branch, after clean passed.
    ui.merge_btn
        .set_sensitive(!running && merge_branch && merge_tool && ui.prep_clean_ok.get());
    // Prep: after clean (and, on the merge branch, merge) passed.
    let ready_for_prep = ui.prep_clean_ok.get() && (!merge_branch || ui.prep_merge_ok.get());
    ui.prep_btn
        .set_sensitive(!running && prep_tool && ready_for_prep);
}

/// Generic staged-prep job runner: stream a single prep command, gate on its
/// exit + a required output artifact, and invoke `on_pass` only when the gate
/// passes. The engine on-done path is NOT used — a prep run never sets Run Root.
fn run_prep_command(
    ui: &Rc<RunEngine>,
    cmd: PrismCommand,
    phase: &'static str,
    chip: gtk::Label,
    needs: PathBuf,
    on_pass: Rc<dyn Fn(&Rc<RunEngine>)>,
) {
    if ui.running.get() {
        return;
    }
    ui.running.set(true);
    ui.cancelled.set(false);
    set_chip(&chip, &format!("{phase}: running"), "info");
    append_log(
        &ui.log_view,
        &format!("\u{25B6}  {}", cmd.display(&ui.ctx.resolver)),
    );
    update_launch_sensitivity(ui);
    refresh_prep_sensitivity(ui);

    let ui_evt = ui.clone();
    let chip_evt = chip.clone();
    let phase_s = phase.to_string();
    let handle = ui
        .ctx
        .jobs
        .spawn_cancellable(cmd, &ui.ctx.resolver, move |ev| match ev {
            JobEvent::Stdout(line) => append_log(&ui_evt.log_view, &line),
            JobEvent::Stderr(line) => append_log(&ui_evt.log_view, &format!("[stderr] {line}")),
            JobEvent::Done(outcome) => {
                ui_evt.running.set(false);
                *ui_evt.job.borrow_mut() = None;
                let code = outcome.code;
                if ui_evt.cancelled.get() {
                    set_chip(&chip_evt, &format!("{phase_s}: cancelled"), "warn");
                    append_log(
                        &ui_evt.log_view,
                        &format!("\u{2717}  {phase_s} cancelled by operator"),
                    );
                } else if matches!(code, Some(0)) && needs.exists() {
                    set_chip(&chip_evt, &format!("{phase_s}: PASS"), "pass");
                    on_pass(&ui_evt);
                } else {
                    let reason = if !matches!(code, Some(0)) {
                        format!(
                            "exit {}",
                            code.map(|c| c.to_string())
                                .unwrap_or_else(|| "signal".into())
                        )
                    } else {
                        format!("expected output missing: {}", needs.display())
                    };
                    set_chip(&chip_evt, &format!("{phase_s}: FAIL"), "blocked");
                    append_log(
                        &ui_evt.log_view,
                        &format!("\u{2717}  {phase_s} failed \u{2014} {reason}"),
                    );
                }
                update_launch_sensitivity(&ui_evt);
                refresh_prep_sensitivity(&ui_evt);
            }
        });
    *ui.job.borrow_mut() = Some(handle);
}

/// Phase B — Clean. Single-chain: one `prism-clean.py` job. Merge branch: clean
/// every chain sequentially (operator initiated the whole phase).
fn run_clean_phase(ui: &Rc<RunEngine>) {
    let raw = match ui.raw_pdb.borrow().clone() {
        Some(r) => r,
        None => return,
    };
    ui.prep_clean_ok.set(false);
    if is_merge_branch(ui) {
        run_clean_chain(ui, raw, 0);
    } else {
        let chains = ui.chains.borrow().clone();
        let chain = chains.first().cloned().unwrap_or_else(|| "A".to_string());
        let stem = pdb_stem(&raw);
        let clean = prep_dir(&raw).join(format!("{stem}_clean.pdb"));
        let cmd = PrismCommand::clean_pdb(&raw, &clean, &chain);
        let clean_stored = clean.clone();
        run_prep_command(
            ui,
            cmd,
            "Clean",
            ui.clean_chip.clone(),
            clean.clone(),
            Rc::new(move |ui: &Rc<RunEngine>| {
                *ui.cleaned_pdb.borrow_mut() = Some(clean_stored.clone());
                ui.prep_clean_ok.set(true);
                ui.prep_phase.set(PrepPhase::NeedPrep);
                refresh_prep_sensitivity(ui);
            }),
        );
    }
}

/// Merge-branch clean driver: clean `chains[idx]`, then chain to the next index
/// on PASS. When all chains are cleaned, unlock the merge phase.
fn run_clean_chain(ui: &Rc<RunEngine>, raw: PathBuf, idx: usize) {
    let chains = ui.chains.borrow().clone();
    if idx >= chains.len() {
        ui.prep_clean_ok.set(true);
        set_chip(&ui.clean_chip, "Clean: PASS (all chains)", "pass");
        ui.prep_phase.set(PrepPhase::NeedMerge);
        refresh_prep_sensitivity(ui);
        refresh_command(ui);
        return;
    }
    let chain = chains[idx].clone();
    let stem = pdb_stem(&raw);
    let out = prep_dir(&raw).join(format!("{stem}_chain{chain}.pdb"));
    let cmd = PrismCommand::clean_pdb(&raw, &out, &chain);
    let raw_next = raw.clone();
    run_prep_command(
        ui,
        cmd,
        "Clean",
        ui.clean_chip.clone(),
        out.clone(),
        Rc::new(move |ui: &Rc<RunEngine>| {
            run_clean_chain(ui, raw_next.clone(), idx + 1);
        }),
    );
}

/// Phase B1/B2 — Merge (multichain MERGE branch only).
fn run_merge_phase(ui: &Rc<RunEngine>) {
    let raw = match ui.raw_pdb.borrow().clone() {
        Some(r) => r,
        None => return,
    };
    let chains = ui.chains.borrow().clone();
    if chains.len() < 2 {
        return;
    }
    let stem = pdb_stem(&raw);
    let dir = prep_dir(&raw);
    let chain_pdbs: Vec<PathBuf> = chains
        .iter()
        .map(|c| dir.join(format!("{stem}_chain{c}.pdb")))
        .collect();
    let merged = dir.join(format!("{stem}_merged.pdb"));
    let map = dir.join(format!("{stem}_merged.chain_map.json"));
    let cmd = PrismCommand::merge_chains(&chain_pdbs, &merged, &map);
    let merged_stored = merged.clone();
    let map_stored = map.clone();
    run_prep_command(
        ui,
        cmd,
        "Merge",
        ui.merge_chip.clone(),
        merged.clone(),
        Rc::new(move |ui: &Rc<RunEngine>| {
            *ui.merged_pdb.borrow_mut() = Some(merged_stored.clone());
            *ui.chain_map.borrow_mut() = Some(map_stored.clone());
            ui.prep_merge_ok.set(true);
            ui.prep_phase.set(PrepPhase::NeedPrep);
            refresh_prep_sensitivity(ui);
            refresh_command(ui);
        }),
    );
}

/// Phase C — Prep topology. Consumes the cleaned (or merged) PDB and, on PASS,
/// funnels the produced `*.topology.json` into `set_topology` (the single
/// launch path). A merge's chain map is already stashed and will be appended.
fn run_prep_phase(ui: &Rc<RunEngine>) {
    let raw = match ui.raw_pdb.borrow().clone() {
        Some(r) => r,
        None => return,
    };
    let merge_branch = is_merge_branch(ui);
    let stem = pdb_stem(&raw);
    let dir = prep_dir(&raw);

    let (input_pdb, topo_out) = if merge_branch {
        let input = match ui.merged_pdb.borrow().clone() {
            Some(m) => m,
            None => return,
        };
        (input, dir.join(format!("{stem}_merged.topology.json")))
    } else {
        let input = match ui.cleaned_pdb.borrow().clone() {
            Some(c) => c,
            None => return,
        };
        (input, dir.join(format!("{stem}_clean.topology.json")))
    };

    let cmd = PrismCommand::prep_topology(
        &input_pdb,
        &topo_out,
        Some("cryptic"),
        false,
        false,
        false,
        None,
        None,
    );
    let topo_stored = topo_out.clone();
    run_prep_command(
        ui,
        cmd,
        "Prep",
        ui.prep_chip.clone(),
        topo_out.clone(),
        Rc::new(move |ui: &Rc<RunEngine>| {
            ui.prep_prep_ok.set(true);
            ui.prep_phase.set(PrepPhase::Ready);
            // Single funnel into the canonical launch flow.
            set_topology(ui, topo_stored.clone());
            append_log(
                &ui.log_view,
                "\u{2713}  prep complete \u{2014} topology loaded; review the command and Launch.",
            );
        }),
    );
}

/// Rebuild the exact-command preview and refresh Launch sensitivity.
fn refresh_command(ui: &Rc<RunEngine>) {
    if ui.primary_manifest.is_some() {
        match build_command(ui) {
            Some(cmd) => ui.cmd_preview.set_text(&cmd.display(&ui.ctx.resolver)),
            None => ui.cmd_preview.set_text("Choose an explicitly prepared 1010 experiment; topology/output fields are not used."),
        }
        update_launch_sensitivity(ui);
        return;
    }
    if let Some(error) = ui.engine_argv_error.borrow().as_ref() {
        ui.cmd_preview
            .set_text(&format!("Engine argv is not launchable: {error}"));
        update_launch_sensitivity(ui);
        return;
    }
    match build_command(ui) {
        Some(cmd) => ui.cmd_preview.set_text(&cmd.display(&ui.ctx.resolver)),
        None => ui.cmd_preview.set_text(
            "Choose a topology and output directory to see the exact governed run command.",
        ),
    }
    update_launch_sensitivity(ui);
}

/// Collect the validated [`SurfaceFlag`] tail from current widget state. Threaded
/// into the C-canonical preset, the C-twin-graph preset, and any first-class named
/// `--engine-surface` run. The twin arm takes the FULL tail (coupling is orthogonal
/// to clustering / enhanced sampling); the named-surface arm drops the backend-
/// selection flags it already owns (see [`SurfaceFlag::owned_by_named_surface`]) and
/// keeps the additive ghost tuning knobs. Smoke presets ignore it.
fn surface_flags(ui: &Rc<RunEngine>) -> Vec<SurfaceFlag> {
    let mut v = Vec::new();
    if ui.rest2.get() {
        v.push(SurfaceFlag::Rest2);
    }
    if ui.multi_temp.get() {
        v.push(SurfaceFlag::MultiTemp);
    }
    if ui.stepped_holds.get() {
        v.push(SurfaceFlag::SteppedHolds);
    }
    // Ghost-phase-lattice tail is a pure decision over the raw widget + surface
    // state; extracted so it is unit-testable without a live GTK RunEngine.
    let backend_ref = ui.clustering_backend.borrow();
    v.extend(ghost_surface_flags(
        backend_ref.as_deref(),
        ui.engine_surface.get(),
        ui.ghost_so3.get(),
        ui.ghost_cell_a.get(),
        ui.ghost_max_temporal.get(),
        ui.ghost_step_bucket.get(),
    ));
    // NOTE: spike-percentile is NOT emitted here. The widget value is threaded
    // straight into validate_and_run_ext's canonical base (build_command), so
    // exactly ONE --spike-percentile token is emitted; pushing it here too would
    // duplicate the flag.
    v
}

/// Push the four `--ghost-phase-lattice-*` tuning knobs. Kept separate so both
/// the "un-named path drives the backend via the widget" branch and the "named
/// `ghost-tiles` surface owns the backend" branch emit an identical knob set.
fn push_ghost_knobs(
    v: &mut Vec<SurfaceFlag>,
    so3: f32,
    cell_a: f32,
    max_temporal: u64,
    step_bucket: u64,
) {
    v.push(SurfaceFlag::GhostSo3(so3));
    v.push(SurfaceFlag::GhostCellA(cell_a));
    v.push(SurfaceFlag::GhostMaxTemporalEdge(max_temporal));
    v.push(SurfaceFlag::GhostStepBucket(step_bucket));
}

/// Pure decision for the ghost-phase-lattice / clustering tail. The 4D lattice
/// knobs are meaningful only when the ghost-phase-lattice-4d backend is actually
/// ACTIVE, which happens two ways:
///   1. Un-named path (`engine_surface == None`, i.e. Canonical/TwinGraph): the
///      operator drives the backend via the clustering-backend widget, so emit
///      the FULL bundle — the backend hard-errors without its two prerequisites
///      (m1-monolithic-discovery + mar-v2-telemetry), so they are unconditional.
///   2. Named `ghost-tiles` surface: the engine expands the backend + both
///      prerequisites UPSTREAM (nhs_rt_full.rs:3691-3695), so we layer ONLY the
///      additive tuning knobs — never the backend-selection flags, which would
///      duplicate the surface's own expansion. This is the gap-(c) hardening: the
///      knobs now survive under `ghost-tiles` even if the widget is left unset.
/// Under any OTHER named surface the ghost lattice is not activated, so a ghost
/// widget selection is ignored entirely (no orphan knobs, no backend). A non-ghost
/// widget backend on the un-named path emits just its single `--clustering-backend`.
fn ghost_surface_flags(
    widget_backend: Option<&str>,
    engine_surface: EngineSurface,
    so3: f32,
    cell_a: f32,
    max_temporal: u64,
    step_bucket: u64,
) -> Vec<SurfaceFlag> {
    let mut v = Vec::new();
    let widget_is_ghost = widget_backend == Some("ghost-phase-lattice-4d");
    if widget_is_ghost && engine_surface == EngineSurface::None {
        v.push(SurfaceFlag::M1MonolithicDiscovery);
        v.push(SurfaceFlag::MarV2Telemetry);
        if let Some(cb) = SurfaceFlag::clustering_backend("ghost-phase-lattice-4d") {
            v.push(cb);
        }
        push_ghost_knobs(&mut v, so3, cell_a, max_temporal, step_bucket);
    } else if engine_surface == EngineSurface::GhostTiles {
        push_ghost_knobs(&mut v, so3, cell_a, max_temporal, step_bucket);
    } else if engine_surface == EngineSurface::None {
        if let Some(backend) = widget_backend {
            if let Some(cb) = SurfaceFlag::clustering_backend(backend) {
                v.push(cb);
            }
        }
    }
    v
}

/// Build the exact `PrismCommand` for the current preset + surface state. This
/// is the SINGLE source of truth shared by the preview and by [`launch`], so
/// what the operator sees is byte-for-byte what fires.
fn build_command(ui: &Rc<RunEngine>) -> Option<PrismCommand> {
    if ui.primary_manifest.is_some() {
        return ui.primary_panel.borrow().as_deref().map(PrismCommand::primary_1010_cycle);
    }
    let topo = ui.topo.borrow().clone()?;
    let output = ui.output.borrow().clone()?;
    let chain_map = ui.chain_map.borrow().clone();
    let cm = chain_map.as_deref();
    // Complete argv is an explicit command class, not a tail on a calibrated
    // preset. It therefore takes precedence over consensus/preset controls and
    // forwards every operator token through the governed adapter unchanged.
    if ui.preset.get() == Preset::CompleteArgv {
        return Some(PrismCommand::validate_and_run_raw(
            &topo,
            &output,
            cm,
            &ui.engine_argv.borrow(),
        ));
    }
    // N-replicate consensus (production) short-circuits every single-run control: it
    // drives scripts/prism_replicate.py, which runs N wrapper passes internally then
    // consensus-clusters the metastable pockets (CLAUDE.md canonical production entry
    // point). Target name is derived from the topology stem.
    if ui.consensus_mode.get() {
        let target = topology_prefix(&topo);
        return Some(PrismCommand::replicate_consensus(
            &topo,
            &target,
            ui.n_replicates.get(),
            &output,
        ));
    }
    let ms = ui.multi_stream.get();
    let mw = ui.max_wall.get();
    // Fundamental physics/capture/detection knobs from the three parameter tabs;
    // threaded into BOTH single-run paths below (never surface-overridden). The
    // protocol flag selects Fast (calibrated) vs Standard (which honors
    // temperature/cryo/steps 1:1).
    let params = RunParams {
        standard_protocol: ui.standard_protocol.get(),
        steps: ui.steps.get(),
        temperature: ui.temperature.get(),
        cryo_temp: ui.cryo_temp.get(),
        save_trajectory_interval: ui.save_traj_interval.get(),
        lining_cutoff: ui.lining_cutoff.get(),
    };
    // A first-class NAMED engine surface takes precedence over the command-class
    // preset AND the manual advanced-surface tail: it is a self-contained
    // expansion the engine performs upstream of every guard/dispatch, so the UI
    // emits ONLY the single --engine-surface token (on a base that drops
    // --multi-differential). The command preview shows exactly this.
    let es = ui.engine_surface.get();
    if es != EngineSurface::None {
        // Gap-(c) fix: layer the operator's advanced-surface tail on top of the
        // named surface. `validate_and_run_engine_surface` filters out the
        // backend-selection flags the surface already owns, so the additive ghost
        // tuning knobs (and REST2/multi-temp/stepped-holds) survive instead of
        // being silently dropped by this early return.
        return Some(PrismCommand::validate_and_run_engine_surface(
            &topo,
            &output,
            ms,
            mw,
            cm,
            es,
            &params,
            &surface_flags(ui),
        ));
    }
    Some(match ui.preset.get() {
        Preset::Canonical => PrismCommand::validate_and_run_ext(
            &topo,
            &output,
            ms,
            mw,
            cm,
            &surface_flags(ui),
            ui.spike_percentile.get(),
            &params,
        ),
        Preset::TwinGraph => {
            // Gap-(c) fix: thread the full surface tail so TWIN can run WITH a
            // tuned ghost lattice / enhanced sampling instead of dropping it.
            PrismCommand::validate_and_run_twin(&topo, &output, ms, mw, cm, &surface_flags(ui))
        }
        Preset::SmokeMin => PrismCommand::smoke_min(&topo, &output),
        Preset::SmokeEng => PrismCommand::smoke_eng(&topo, &output),
        Preset::CompleteArgv => unreachable!("complete argv returned above"),
    })
}

/// Launch is enabled only when the script resolves, a topology + output are set,
/// and no run is in flight. Cancel is enabled only while a run is in flight.
fn update_launch_sensitivity(ui: &Rc<RunEngine>) {
    if ui.primary_manifest.is_some() {
        ui.launch_btn.set_sensitive(ui.script_ok.get() && ui.primary_panel_verified.get() && !ui.running.get());
        ui.cancel_btn.set_sensitive(ui.running.get());
        return;
    }
    let ready = ui.script_ok.get()
        && ui.topo.borrow().is_some()
        && ui.output.borrow().is_some()
        && ui.engine_argv_error.borrow().is_none()
        && !ui.running.get();
    ui.launch_btn.set_sensitive(ready);
    ui.cancel_btn.set_sensitive(ui.running.get());
}

fn inspect_primary_panel(ui: &Rc<RunEngine>, path: PathBuf) {
    if ui.running.get() { return; }
    *ui.primary_panel.borrow_mut() = Some(path.clone());
    ui.primary_panel_verified.set(false);
    ui.primary_panel_label.set_text(&format!("Inspecting prepared experiment: {}", path.display()));
    refresh_command(ui);
    let report_ok = Rc::new(Cell::new(false));
    let ok = report_ok.clone();
    let ui_evt = ui.clone();
    ui.ctx.jobs.spawn(PrismCommand::primary_1010_inspect(&path), &ui.ctx.resolver, move |event| {
        match event {
            JobEvent::Stdout(line) => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                    ok.set(value["schema"] == "prism.primary1010_prepared_panel_inspection.v1"
                        && value["status"] == "IMPLEMENTATION_VERIFIED"
                        && value["panel_root"].as_str().map(Path::new) == Some(path.as_path()));
                }
            }
            JobEvent::Stderr(line) => append_log(&ui_evt.log_view, &line),
            JobEvent::Done(result) => {
                if ui_evt.primary_panel.borrow().as_ref() != Some(&path) { return; }
                let valid = result.code == Some(0) && ok.get();
                ui_evt.primary_panel_verified.set(valid);
                ui_evt.primary_panel_label.set_text(&format!("{}: {}", if valid { "Prepared bindings verified" } else { "Prepared experiment rejected; no native launch" }, path.display()));
                refresh_command(&ui_evt);
            }
        }
    });
}

fn launch_primary(ui: &Rc<RunEngine>) {
    if ui.running.get() || !ui.script_ok.get() || !ui.primary_panel_verified.get() { return; }
    let panel = match ui.primary_panel.borrow().clone() { Some(p) => p, None => return };
    let cmd = match build_command(ui) { Some(c) => c, None => return };
    ui.running.set(true);
    ui.cancelled.set(false);
    ui.primary_panel_verified.set(false); // A campaign cannot be re-fired from the same UI selection.
    while let Some(child) = ui.primary_results.first_child() { ui.primary_results.remove(&child); }
    ui.log_view.buffer().set_text("");
    append_log(&ui.log_view, &format!("Launching exact production request: {}", cmd.display(&ui.ctx.resolver)));
    set_chip(&ui.preflight_chip, "Prepared1010: verified", "pass");
    set_chip(&ui.engine_chip, "PathA → PathB: running", "info");
    set_chip(&ui.postflight_chip, "Scoring/archive: pending", "info");
    update_launch_sensitivity(ui);
    let ui_evt = ui.clone();
    let handle = ui.ctx.jobs.spawn_controller(cmd, &ui.ctx.resolver, move |event| match event {
        JobEvent::Stdout(line) => append_log(&ui_evt.log_view, &line),
        JobEvent::Stderr(line) => append_log(&ui_evt.log_view, &format!("[stderr] {line}")),
        JobEvent::Done(outcome) => {
            *ui_evt.job.borrow_mut() = None;
            if ui_evt.cancelled.get() || outcome.code != Some(0) {
                ui_evt.running.set(false);
                set_chip(&ui_evt.engine_chip, "Workflow incomplete", "blocked");
                append_log(&ui_evt.log_view, "Controller did not complete; scientific output, partial checkpoints and spent budget remain authoritative. Run Root unchanged.");
                update_launch_sensitivity(&ui_evt);
            } else { verify_primary_result(&ui_evt, panel.clone()); }
        }
    });
    *ui.job.borrow_mut() = Some(handle);
}

fn verify_primary_result(ui: &Rc<RunEngine>, panel: PathBuf) {
    let actual = Rc::new(RefCell::new(None::<serde_json::Value>));
    let saved = actual.clone();
    let ui_evt = ui.clone();
    ui.ctx.jobs.spawn(PrismCommand::primary_1010_result(&panel), &ui.ctx.resolver, move |event| match event {
        JobEvent::Stdout(line) => {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                if value["schema"] == "prism.primary1010_actual_cycle_completion.v1"
                    && value["status"] == "IMPLEMENTATION_VERIFIED"
                    && value["panel_root"].as_str().map(Path::new) == Some(panel.as_path()) {
                    *saved.borrow_mut() = Some(value);
                }
            }
        }
        JobEvent::Stderr(line) => append_log(&ui_evt.log_view, &line),
        JobEvent::Done(outcome) => {
            ui_evt.running.set(false);
            let report = saved.borrow();
            let rows = report.as_ref().and_then(|v| v["per_target_archives"].as_array());
            let valid = outcome.code == Some(0) && !ui_evt.cancelled.get()
                && rows.map(|r| r.len() == 4).unwrap_or(false);
            if valid {
                for row in rows.unwrap() {
                    if let (Some(target), Some(archive)) = (row["target"].as_str(), row["archive"].as_str()) {
                        let button = gtk::Button::with_label(&format!("Open {target} results"));
                        let ctx = ui_evt.ctx.clone();
                        let path = PathBuf::from(archive);
                        button.connect_clicked(move |_| ctx.set_run_dir(Some(path.clone())));
                        ui_evt.primary_results.append(&button);
                    }
                }
                set_chip(&ui_evt.engine_chip, "PathA/PathB: PASS", "pass");
                set_chip(&ui_evt.postflight_chip, "Four targets/scorer/archive: PASS", "pass");
                append_log(&ui_evt.log_view, "Complete production result verified. Choose a per-target archive to open its scientific results; no target or rank has been substituted.");
            } else {
                set_chip(&ui_evt.postflight_chip, "Result verification failed", "blocked");
                append_log(&ui_evt.log_view, "No verified complete result; Run Root unchanged. Partial data and original error remain retained.");
            }
            update_launch_sensitivity(&ui_evt);
        }
    });
}

/// Start the governed run. The operator clicked Launch; this never auto-fires.
fn launch(ui: &Rc<RunEngine>) {
    if ui.primary_manifest.is_some() { launch_primary(ui); return; }
    if ui.running.get() || !ui.script_ok.get() {
        return;
    }
    let topo = match ui.topo.borrow().clone() {
        Some(t) => t,
        None => return,
    };
    let output = match ui.output.borrow().clone() {
        Some(o) => o,
        None => return,
    };
    match crate::adapters::governed_output_for_topology(&topo) {
        Ok(governed) if governed == output => {}
        Ok(governed) => {
            append_log(&ui.log_view, &serde_json::json!({
                "status":"rejected", "reason":"OUTPUT_POLICY_CONFLICT",
                "requested_output":output, "governed_output":governed
            }).to_string());
            return;
        }
        Err(error) => {
            append_log(&ui.log_view, &serde_json::json!({
                "status":"rejected", "reason":"OUTPUT_POLICY_UNAVAILABLE", "detail":error
            }).to_string());
            return;
        }
    }
    let cmd = match build_command(ui) {
        Some(c) => c,
        None => return,
    };
    // Freeze the evidence contract for this process. Controls remain responsive
    // while output streams, so completion must not consult later UI state.
    ui.active_consensus_required.set(cmd.bin == PrismBin::PrismReplicate);
    ui.active_md_only_required.set(
        ui.preset.get() != Preset::CompleteArgv
            || ui
                .engine_argv
                .borrow()
                .tokens()
                .iter()
                .any(|token| token == "--md-only-evidence"),
    );

    // Reset lifecycle + gate state.
    ui.running.set(true);
    ui.cancelled.set(false);
    ui.postflight_pass.set(None);
    set_chip(&ui.preflight_chip, "Preflight: running", "info");
    set_chip(&ui.engine_chip, "Engine: pending", "info");
    set_chip(&ui.postflight_chip, "Postflight: pending", "info");
    ui.log_view.buffer().set_text("");
    update_launch_sensitivity(ui);

    let _ = &topo;
    append_log(&ui.log_view, "\u{25B6}  launching governed run\u{2026}");

    let ui_evt = ui.clone();
    let out_evt = output.clone();
    let handle = ui
        .ctx
        .jobs
        .spawn_cancellable(cmd, &ui.ctx.resolver, move |ev| match ev {
            JobEvent::Stdout(line) => on_line(&ui_evt, &line),
            JobEvent::Stderr(line) => on_line(&ui_evt, &format!("[stderr] {line}")),
            JobEvent::Done(outcome) => on_done(&ui_evt, &out_evt, outcome.code),
        });
    *ui.job.borrow_mut() = Some(handle);
}

/// Cancel: SIGTERM the whole process group now, escalate to SIGKILL after a
/// grace period if it is still alive. A cancelled run NEVER sets the run root.
fn cancel(ui: &Rc<RunEngine>) {
    let handle = ui.job.borrow().clone();
    if let Some(handle) = handle {
        ui.cancelled.set(true);
        append_log(
            &ui.log_view,
            "\u{25FC}  cancel requested \u{2014} sending SIGTERM to the process group\u{2026}",
        );
        handle.cancel();
        set_chip(&ui.engine_chip, "Engine: cancelling", "warn");
        if ui.primary_manifest.is_some() {
            append_log(&ui.log_view, "Waiting for the production controller’s durable stop, native/MPS cleanup and C1/lease closure. UI does not kill workers or grant another budget.");
            return;
        }
        // Escalate to SIGKILL if the tree has not exited within the grace window.
        let handle_kill = handle.clone();
        let log = ui.log_view.clone();
        glib::timeout_add_local_once(Duration::from_secs(8), move || {
            if handle_kill.is_running() {
                append_log(
                    &log,
                    "\u{25FC}  still running after grace \u{2014} sending SIGKILL",
                );
                handle_kill.kill();
            }
        });
    }
}

/// Handle one streamed output line: append to the log and update gate chips off
/// STABLE banner tokens emitted by the script / preflight / postflight.
fn on_line(ui: &Rc<RunEngine>, line: &str) {
    append_log(&ui.log_view, line);
    if line.contains("PREFLIGHT: PASS") {
        set_chip(&ui.preflight_chip, "Preflight: PASS", "pass");
    } else if line.contains("PREFLIGHT: FAIL") || line.contains("PREFLIGHT FAILED") {
        set_chip(&ui.preflight_chip, "Preflight: FAIL", "blocked");
    }
    if line.contains("RUNNING PRISM-4D ENGINE") {
        set_chip(&ui.engine_chip, "Engine: running", "info");
    }
    if line.contains("POSTFLIGHT_MD_ONLY: PASS") || line.contains("POSTFLIGHT: PASS") {
        ui.postflight_pass.set(Some(true));
        set_chip(&ui.postflight_chip, "Postflight: PASS", "pass");
    } else if line.contains("POSTFLIGHT_MD_ONLY: FAIL") || line.contains("POSTFLIGHT: FAIL") {
        ui.postflight_pass.set(Some(false));
        set_chip(&ui.postflight_chip, "Postflight: FAIL", "blocked");
    }
}

/// Terminal handling: decide success ourselves (never trust exit==0), and set
/// the app Run Root only on a genuinely passing, non-cancelled run.
fn on_done(ui: &Rc<RunEngine>, output: &Path, code: Option<i32>) {
    ui.running.set(false);
    *ui.job.borrow_mut() = None;
    update_launch_sensitivity(ui);

    // Crash-class exits are never promoted to success without a typed,
    // independently verified commit-last receipt exception (none exists).
    let engine_ok = matches!(code, Some(0));
    if engine_ok {
        set_chip(&ui.engine_chip, "Engine: exited (ok)", "pass");
    } else {
        let shown = code
            .map(|c| c.to_string())
            .unwrap_or_else(|| "signal".into());
        set_chip(&ui.engine_chip, &format!("Engine: exit {shown}"), "blocked");
    }

    if ui.cancelled.get() {
        append_log(
            &ui.log_view,
            "\u{2717}  run cancelled by operator \u{2014} Run Root left UNCHANGED (partial output).",
        );
        ui.ctx.toast("Run cancelled — Run Root unchanged");
        return;
    }

    if ui.active_consensus_required.get() {
        let verified = std::process::Command::new("/usr/bin/python3")
            .arg(ui.ctx.resolver.resolve(PrismBin::PrismReplicate))
            .arg("--verify-completion")
            .arg(output.join("PRISMOS_REPLICATE_COMPLETION.json"))
            .output();
        match verified {
            Ok(result) if engine_ok && result.status.success() => {
                let report = serde_json::from_slice::<serde_json::Value>(&result.stdout).ok();
                if report.as_ref().and_then(|value| value.get("status")).and_then(serde_json::Value::as_str) == Some("PASS") {
                    set_chip(&ui.postflight_chip, "Replicates: PASS", "pass");
                    ui.ctx.set_run_dir(Some(output.to_path_buf()));
                    append_log(&ui.log_view, &String::from_utf8_lossy(&result.stdout));
                    return;
                }
                append_log(&ui.log_view, "REPLICATE_COMPLETION_INVALID: verifier did not return PASS");
            }
            Ok(result) => append_log(&ui.log_view, &format!("REPLICATE_COMPLETION_REJECTED: {}", String::from_utf8_lossy(&result.stderr))),
            Err(error) => append_log(&ui.log_view, &format!("REPLICATE_COMPLETION_UNAVAILABLE: {error}")),
        }
        set_chip(&ui.postflight_chip, "Replicates: rejected", "blocked");
        return;
    }

    let postflight_pass = ui.postflight_pass.get() == Some(true);
    let manifest = output.join("md_evidence_manifest.json");
    let spikes = read_total_spikes_md(&manifest);
    let spikes_ok = spikes.map(|n| n > 0).unwrap_or(false);
    let md_only_required = ui.active_md_only_required.get();
    let evidence_ok = !md_only_required || spikes_ok;

    if engine_ok && postflight_pass && evidence_ok {
        ui.ctx.set_run_dir(Some(output.to_path_buf()));
        let acceptance = if md_only_required {
            format!(
                "\u{2713}  RUN PASSED \u{2014} engine ok, POSTFLIGHT_MD_ONLY: PASS, \
                 total_spikes_md={}. Run Root set to {}.",
                spikes.unwrap_or(0),
                output.display()
            )
        } else {
            format!(
                "\u{2713}  RUN PASSED \u{2014} engine ok, governed POSTFLIGHT: PASS. \
                 Run Root set to {}.",
                output.display()
            )
        };
        append_log(&ui.log_view, &acceptance);
        append_log(
            &ui.log_view,
            "   Reselect the science panels (Binding Sites, Scientific Results, \u{2026}) to load \
             this run \u{2014} already-built panels do not hot-reload.",
        );
        ui.ctx
            .toast("Run passed — Run Root updated (reselect science panels to load)");
    } else {
        // Honest, specific failure explanation — do NOT set the run root.
        let mut reasons = Vec::new();
        if !engine_ok {
            reasons.push(format!(
                "engine exit {}",
                code.map(|c| c.to_string())
                    .unwrap_or_else(|| "signal".into())
            ));
        }
        if !postflight_pass {
            reasons.push("no governed postflight PASS".to_string());
        }
        if md_only_required && !spikes_ok {
            match spikes {
                Some(n) => reasons.push(format!("total_spikes_md={n} (needs > 0)")),
                None => reasons.push(format!(
                    "could not read total_spikes_md from {}",
                    manifest.display()
                )),
            }
        }
        append_log(
            &ui.log_view,
            &format!(
                "\u{2717}  RUN DID NOT PASS GATES \u{2014} Run Root UNCHANGED. Reasons: {}.",
                reasons.join("; ")
            ),
        );
        ui.ctx.toast("Run did not pass gates — Run Root unchanged");
    }
}

/// Read `total_spikes_md` from an md-only evidence manifest.
fn read_total_spikes_md(path: &Path) -> Option<i64> {
    let text = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("total_spikes_md").and_then(|x| x.as_i64())
}

/// Append a line to the live log and keep the view scrolled to the bottom.
fn append_log(view: &gtk::TextView, line: &str) {
    let buf = view.buffer();
    let mut end = buf.end_iter();
    buf.insert(&mut end, line);
    buf.insert(&mut end, "\n");
    let mark = buf.create_mark(None, &buf.end_iter(), false);
    view.scroll_to_mark(&mark, 0.0, false, 0.0, 0.0);
}

/// Build a status-pill gate chip.
fn make_chip(text: &str) -> gtk::Label {
    let chip = gtk::Label::new(Some(text));
    chip.add_css_class("status-pill");
    chip.add_css_class("gate-chip");
    chip.add_css_class("info");
    chip.set_valign(gtk::Align::Center);
    chip
}

/// Set a gate chip's text and status class (`info` / `pass` / `warn` /
/// `blocked`), clearing any prior status class.
fn set_chip(chip: &gtk::Label, text: &str, state: &str) {
    chip.set_text(text);
    for c in ["info", "pass", "warn", "blocked"] {
        chip.remove_css_class(c);
    }
    chip.add_css_class(state);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_surface_index_maps_to_named_surfaces() {
        assert_eq!(engine_surface_from_index(0), EngineSurface::None);
        assert_eq!(
            engine_surface_from_index(1),
            EngineSurface::CapturedGraphTwin
        );
        assert_eq!(
            engine_surface_from_index(2),
            EngineSurface::InterferometricTwin
        );
        assert_eq!(engine_surface_from_index(3), EngineSurface::Twin16);
        assert_eq!(engine_surface_from_index(4), EngineSurface::GhostTiles);
        // Out-of-range falls back to the safe no-op default.
        assert_eq!(engine_surface_from_index(99), EngineSurface::None);
    }

    #[test]
    fn engine_surface_labels_are_index_aligned_and_complete() {
        // One label per selectable index; index 0 is the no-op "none".
        assert_eq!(ENGINE_SURFACE_LABELS.len(), 5);
        for (i, _) in ENGINE_SURFACE_LABELS.iter().enumerate() {
            // Every label index resolves to a defined EngineSurface (no panic).
            let _ = engine_surface_from_index(i as u32);
        }
        assert_eq!(engine_surface_from_index(0), EngineSurface::None);
    }

    #[test]
    fn named_surface_command_emits_engine_surface_and_drops_multi_differential() {
        // The panel's build path (validate_and_run_engine_surface) emits the
        // single named token and never the multi-differential base flag — the
        // correctness contract for captured-graph-twin.
        let cmd = PrismCommand::validate_and_run_engine_surface(
            Path::new("/t.topology.json"),
            Path::new("/o"),
            8,
            180,
            None,
            EngineSurface::CapturedGraphTwin,
            &RunParams::default(),
            &[],
        );
        let es = cmd
            .args
            .iter()
            .position(|a| a == "--engine-surface")
            .unwrap();
        assert_eq!(cmd.args[es + 1].to_str(), Some("captured-graph-twin"));
        assert!(!cmd.args.iter().any(|a| a == "--multi-differential"));
    }

    // ---- gap-(c) hardening: ghost_surface_flags pure decision -----------

    #[test]
    fn ghost_knobs_survive_under_ghost_tiles_surface_without_widget() {
        // Hardening: picking the ghost-tiles NAMED surface tunes the lattice even
        // if the clustering-backend widget is left unset — knobs emitted, backend
        // NOT (the surface owns the backend + its two prerequisites).
        let v = ghost_surface_flags(None, EngineSurface::GhostTiles, 0.8, 5.0, 500, 500);
        assert!(v.iter().any(|f| matches!(f, SurfaceFlag::GhostSo3(_))));
        assert!(v.iter().any(|f| matches!(f, SurfaceFlag::GhostCellA(_))));
        assert!(v
            .iter()
            .any(|f| matches!(f, SurfaceFlag::GhostMaxTemporalEdge(_))));
        assert!(v
            .iter()
            .any(|f| matches!(f, SurfaceFlag::GhostStepBucket(_))));
        assert!(!v
            .iter()
            .any(|f| matches!(f, SurfaceFlag::ClusteringBackend(_))));
        assert!(!v
            .iter()
            .any(|f| matches!(f, SurfaceFlag::M1MonolithicDiscovery)));
        assert!(!v.iter().any(|f| matches!(f, SurfaceFlag::MarV2Telemetry)));
    }

    #[test]
    fn ghost_widget_on_unnamed_path_emits_full_bundle() {
        // Un-named path (Canonical/TwinGraph): the widget drives the backend, so
        // the full bundle (prereqs + backend + knobs) must be emitted.
        let v = ghost_surface_flags(
            Some("ghost-phase-lattice-4d"),
            EngineSurface::None,
            0.8,
            5.0,
            500,
            500,
        );
        assert!(v
            .iter()
            .any(|f| matches!(f, SurfaceFlag::M1MonolithicDiscovery)));
        assert!(v.iter().any(|f| matches!(f, SurfaceFlag::MarV2Telemetry)));
        assert!(v.iter().any(
            |f| matches!(f, SurfaceFlag::ClusteringBackend(b) if b == "ghost-phase-lattice-4d")
        ));
        assert!(v.iter().any(|f| matches!(f, SurfaceFlag::GhostSo3(_))));
    }

    #[test]
    fn ghost_widget_ignored_under_non_ghost_named_surface() {
        // twin16 does not activate the ghost lattice; a ghost widget selection
        // under it is ignored entirely — no orphan knobs, no backend.
        let v = ghost_surface_flags(
            Some("ghost-phase-lattice-4d"),
            EngineSurface::Twin16,
            0.8,
            5.0,
            500,
            500,
        );
        assert!(
            v.is_empty(),
            "ghost widget must be ignored under a non-ghost named surface, got {v:?}"
        );
    }

    #[test]
    fn non_ghost_backend_emitted_on_unnamed_path() {
        let v = ghost_surface_flags(Some("gpu-hash"), EngineSurface::None, 0.8, 5.0, 500, 500);
        assert_eq!(v.len(), 1);
        assert!(matches!(&v[0], SurfaceFlag::ClusteringBackend(b) if b == "gpu-hash"));
    }
}
