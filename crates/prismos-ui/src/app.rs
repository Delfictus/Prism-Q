//! GTK4 + libadwaita application shell: the window, navigation, and the shared
//! [`AppContext`] every panel is built against.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use adw::prelude::*;

use crate::adapters::BinaryResolver;
use crate::jobs::JobRunner;
use crate::panels;

/// Shared application state handed to every panel builder. Cheaply cloneable:
/// GTK objects are reference-counted and the rest is behind `Rc`.
#[derive(Clone)]
pub struct AppContext {
    /// Resolves backend binaries to executable paths.
    pub resolver: Rc<BinaryResolver>,
    /// The async job runner (shared global observability log).
    pub jobs: JobRunner,
    /// The currently selected run root, if any.
    pub run_dir: Rc<RefCell<Option<PathBuf>>>,
    /// Top-level window (transient parent for file dialogs).
    pub window: adw::ApplicationWindow,
    /// Toast overlay for transient notifications.
    pub toasts: adw::ToastOverlay,
}

impl AppContext {
    /// Show a transient toast notification.
    pub fn toast(&self, text: &str) {
        self.toasts.add_toast(adw::Toast::new(text));
    }

    /// The currently selected run root.
    pub fn run_dir(&self) -> Option<PathBuf> {
        self.run_dir.borrow().clone()
    }

    /// Set the current run root.
    pub fn set_run_dir(&self, dir: Option<PathBuf>) {
        *self.run_dir.borrow_mut() = dir;
    }
}

/// Launch the desktop application. Returns the process exit code.
pub fn run() -> i32 {
    let app = adw::Application::builder()
        .application_id(crate::APP_ID)
        .build();

    app.connect_startup(|_| crate::theme::install());
    app.connect_activate(build_window);

    let args: Vec<String> = std::env::args().collect();
    app.run_with_args(&args).into()
}

fn build_window(app: &adw::Application) {
    // Single-instance guard: the app registers a unique application id, so a
    // relaunch re-emits `activate` on the running primary. Re-present the
    // existing window instead of building a second one with divergent state.
    if let Some(existing) = app.active_window() {
        existing.present();
        return;
    }

    let window = adw::ApplicationWindow::builder()
        .application(app)
        .default_width(1320)
        .default_height(860)
        .title(crate::APP_NAME)
        .build();
    window.add_css_class("prismos-root");

    let toasts = adw::ToastOverlay::new();

    let ctx = AppContext {
        resolver: Rc::new(BinaryResolver::with_defaults()),
        jobs: JobRunner::new(),
        run_dir: Rc::new(RefCell::new(default_run_dir())),
        window: window.clone(),
        toasts: toasts.clone(),
    };

    // Content stack — one page per panel.
    let stack = gtk::Stack::new();
    stack.set_transition_type(gtk::StackTransitionType::Crossfade);
    stack.set_hexpand(true);
    stack.set_vexpand(true);

    // Sidebar list — one row per panel.
    let sidebar_list = gtk::ListBox::new();
    sidebar_list.add_css_class("navigation-sidebar");
    sidebar_list.set_selection_mode(gtk::SelectionMode::Single);

    let panel_infos = panels::all();
    let ids: Vec<&'static str> = panel_infos.iter().map(|info| info.id).collect();
    for info in &panel_infos {
        let widget = panels::build(info.id, &ctx);
        stack.add_titled(&widget, Some(info.id), info.title);

        let row = adw::ActionRow::builder().title(info.title).build();
        let icon = gtk::Image::from_icon_name(info.icon);
        row.add_prefix(&icon);
        sidebar_list.append(&row);
    }

    let stack_for_select = stack.clone();
    let ids_for_closure = ids.clone();
    sidebar_list.connect_row_selected(move |_list, row| {
        if let Some(row) = row {
            if let Some(id) = ids_for_closure.get(row.index() as usize) {
                stack_for_select.set_visible_child_name(id);
            }
        }
    });
    // Optional deep-link: PRISMOS_UI_PANEL=<id> opens directly to a panel
    // (used for deterministic testing/screenshots); defaults to the first.
    let initial_index = std::env::var("PRISMOS_UI_PANEL")
        .ok()
        .and_then(|wanted| ids.iter().position(|id| *id == wanted))
        .unwrap_or(0);
    if let Some(row) = sidebar_list.row_at_index(initial_index as i32) {
        sidebar_list.select_row(Some(&row));
    }

    // Sidebar navigation page.
    let sidebar_header = adw::HeaderBar::new();
    sidebar_header.set_title_widget(Some(&adw::WindowTitle::new(crate::APP_NAME, "Beta2.0")));
    let sidebar_scroll = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .child(&sidebar_list)
        .build();
    sidebar_scroll.add_css_class("sidebar-pane");
    let sidebar_toolbar = adw::ToolbarView::new();
    sidebar_toolbar.add_top_bar(&sidebar_header);
    sidebar_toolbar.set_content(Some(&sidebar_scroll));
    let sidebar_page = adw::NavigationPage::builder()
        .title(crate::APP_NAME)
        .child(&sidebar_toolbar)
        .build();

    // Content navigation page.
    let content_header = adw::HeaderBar::new();
    content_header.set_title_widget(Some(&adw::WindowTitle::new("Console", "")));
    let content_toolbar = adw::ToolbarView::new();
    content_toolbar.add_top_bar(&content_header);
    content_toolbar.set_content(Some(&stack));
    let content_page = adw::NavigationPage::builder()
        .title("Console")
        .child(&content_toolbar)
        .build();

    let split = adw::NavigationSplitView::new();
    split.set_sidebar(Some(&sidebar_page));
    split.set_content(Some(&content_page));
    split.set_min_sidebar_width(240.0);
    split.set_max_sidebar_width(320.0);

    toasts.set_child(Some(&split));
    window.set_content(Some(&toasts));
    window.present();
}

/// Default run root: an explicit deployment selection, then installed state.
/// No source-tree or current-working-directory fallback is permitted.
fn default_run_dir() -> Option<PathBuf> {
    if let Ok(configured) = std::env::var("PRISMOS_UI_RUN_ROOT") {
        let path = PathBuf::from(configured);
        if path.is_dir() {
            return Some(path);
        }
    }
    ["/var/lib/prismos/runs/default", "/var/lib/prismos/runs"]
        .into_iter()
        .map(PathBuf::from)
        .find(|path| path.is_dir())
}
