//! Desktop panels. Each panel module exposes `build(ctx: &AppContext) ->
//! gtk::Widget`. The shell registers them via [`all`] and dispatches with
//! [`build`].

use adw::prelude::*;

use crate::app::AppContext;

pub mod binding_sites;
pub mod companion;
pub mod corpus;
pub mod cryptic_sites;
pub mod dashboard;
pub mod diagnostics;
pub mod evidence;
pub mod logs;
pub mod oracle;
pub mod pocket_occupancy;
pub mod qnanl;
pub mod run_browser;
pub mod run_engine;
pub mod scientific_results;
pub mod scorer;

/// Sidebar registration for one panel.
pub struct PanelInfo {
    pub id: &'static str,
    pub title: &'static str,
    pub icon: &'static str,
}

/// All panels in sidebar order.
pub fn all() -> Vec<PanelInfo> {
    vec![
        PanelInfo {
            id: "dashboard",
            title: "Dashboard",
            icon: "utilities-system-monitor-symbolic",
        },
        PanelInfo {
            id: "companion",
            title: "Prism CoPilot",
            icon: "user-available-symbolic",
        },
        PanelInfo {
            id: "run_engine",
            title: "Run Engine",
            icon: "media-playback-start-symbolic",
        },
        // --- DSTW wheelhouse: the physics/state tools the CoPilot calls ---
        PanelInfo {
            id: "oracle",
            title: "PrismAI",
            icon: "system-run-symbolic",
        },
        PanelInfo {
            id: "scorer",
            title: "Prism Affinity",
            icon: "emblem-favorite-symbolic",
        },
        PanelInfo {
            id: "state_graph",
            title: "Scientific State",
            icon: "network-workgroup-symbolic",
        },
        PanelInfo {
            id: "pocket_occupancy",
            title: "Pocket Evolution",
            icon: "media-playlist-consecutive-symbolic",
        },
        PanelInfo {
            id: "run_browser",
            title: "Run Root",
            icon: "folder-symbolic",
        },
        PanelInfo {
            id: "scientific_results",
            title: "Scientific Results",
            icon: "applications-science-symbolic",
        },
        PanelInfo {
            id: "binding_sites",
            title: "Binding Sites",
            icon: "find-location-symbolic",
        },
        PanelInfo {
            id: "cryptic_sites",
            title: "Cryptic Sites",
            icon: "edit-find-symbolic",
        },
        PanelInfo {
            id: "evidence",
            title: "Evidence",
            icon: "dialog-information-symbolic",
        },
        PanelInfo {
            id: "qnanl",
            title: "QNANL",
            icon: "security-high-symbolic",
        },
        PanelInfo {
            id: "corpus",
            title: "Corpus Readiness",
            icon: "view-list-symbolic",
        },
        PanelInfo {
            id: "diagnostics",
            title: "Diagnostics",
            icon: "preferences-system-symbolic",
        },
        PanelInfo {
            id: "logs",
            title: "Logs",
            icon: "text-x-generic-symbolic",
        },
    ]
}

/// Build a panel widget by id.
pub fn build(id: &str, ctx: &AppContext) -> gtk::Widget {
    match id {
        "dashboard" => dashboard::build(ctx),
        "companion" => companion::build(ctx),
        "run_engine" => run_engine::build(ctx),
        "oracle" => oracle::build(ctx),
        "scorer" => scorer::build(ctx),
        "run_browser" => run_browser::build(ctx),
        "state_graph" => crate::viz::state_graph_area::build_panel(ctx),
        "pocket_occupancy" => pocket_occupancy::build(ctx),
        "scientific_results" => scientific_results::build(ctx),
        "binding_sites" => binding_sites::build(ctx),
        "cryptic_sites" => cryptic_sites::build(ctx),
        "evidence" => evidence::build(ctx),
        "qnanl" => qnanl::build(ctx),
        "corpus" => corpus::build(ctx),
        "diagnostics" => diagnostics::build(ctx),
        "logs" => logs::build(ctx),
        _ => gtk::Label::new(Some("unknown panel")).upcast(),
    }
}
