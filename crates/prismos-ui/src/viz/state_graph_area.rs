//! DSTW-T4 scientific-state graph visualization.
//!
//! Renders `PRISM_NATIVE_STATE_GRAPH.jsonl` (the durable T4 binding object) as a
//! left-to-right layered DAG on a custom [`gtk::DrawingArea`]. The load-bearing
//! Beta2.0 requirement is to make the **native-vs-derived** provenance of every
//! state vertex visually obvious: each node is coloured by its
//! [`RuntimeOrigin`](crate::model::state_graph::RuntimeOrigin) — runtime-native
//! emissions in cyan, post-run-derived in gray, external-derived in amber, and
//! blocked states in red — with a legend pinned near the graph origin.
//!
//! Interaction is fully local to the panel: scroll to zoom, drag to pan, and
//! click a node to inspect its full provenance fields in the right-hand detail
//! pane. All backend access goes through the parser only; this surface reads a
//! file and never spawns a process.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::rc::Rc;

use adw::prelude::*;

use crate::app::AppContext;
use crate::model::state_graph::{StateGraph, Vertex};

/// File name of the durable state-graph binding object.
const GRAPH_FILE: &str = "PRISM_NATIVE_STATE_GRAPH.jsonl";

/// Detail-pane field order. Matches the slice produced by [`DetailsPane::show`].
const DETAIL_FIELDS: [&str; 10] = [
    "id",
    "state_object_class",
    "artifact_kind",
    "origin",
    "runtime_native",
    "decode_proof_pass",
    "json_path",
    "qnanl_packet_path",
    "tokenizer_family",
    "artifact_sha256",
];

/// Laid-out screen rectangle for one vertex. `vertex_index` indexes into
/// `StateGraph::vertices`. Coordinates are in graph space (before the
/// pan/zoom transform is applied at draw time).
struct NodeBox {
    vertex_index: usize,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

/// The right-hand provenance inspector. Holds the value labels in
/// [`DETAIL_FIELDS`] order so a node click can repopulate them in place.
struct DetailsPane {
    container: gtk::Box,
    values: Vec<gtk::Label>,
}

impl DetailsPane {
    fn new() -> Self {
        let container = gtk::Box::new(gtk::Orientation::Vertical, 8);
        container.add_css_class("card");
        container.set_margin_top(12);
        container.set_margin_bottom(12);
        container.set_margin_start(12);
        container.set_margin_end(12);

        let title = gtk::Label::new(Some("Vertex Detail"));
        title.add_css_class("prismos-title");
        title.set_xalign(0.0);
        container.append(&title);

        let hint = gtk::Label::new(Some(
            "Click a node to inspect its full PrismOS provenance fields.",
        ));
        hint.add_css_class("dim");
        hint.set_xalign(0.0);
        hint.set_wrap(true);
        container.append(&hint);

        let mut values = Vec::with_capacity(DETAIL_FIELDS.len());
        for field in DETAIL_FIELDS {
            let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);

            let key = gtk::Label::new(Some(field));
            key.add_css_class("dim");
            key.set_xalign(0.0);
            key.set_valign(gtk::Align::Start);
            key.set_width_chars(18);

            let val = gtk::Label::new(Some("\u{2014}"));
            val.add_css_class("mono");
            val.set_xalign(0.0);
            val.set_hexpand(true);
            val.set_wrap(true);
            val.set_max_width_chars(46);
            val.set_selectable(true);

            row.append(&key);
            row.append(&val);
            container.append(&row);
            values.push(val);
        }

        DetailsPane { container, values }
    }

    /// Populate the inspector from a vertex.
    fn show(&self, v: &Vertex) {
        let dash = "\u{2014}".to_string();
        let vals: [String; 10] = [
            v.id.clone(),
            v.state_object_class.clone(),
            v.artifact_kind.clone(),
            v.origin.label(),
            v.runtime_native.to_string(),
            v.decode_proof_pass.to_string(),
            v.json_path.clone().unwrap_or_else(|| dash.clone()),
            v.qnanl_packet_path.clone().unwrap_or_else(|| dash.clone()),
            v.tokenizer_family.clone().unwrap_or_else(|| dash.clone()),
            v.artifact_sha256.clone().unwrap_or_else(|| dash.clone()),
        ];
        for (lbl, val) in self.values.iter().zip(vals.iter()) {
            lbl.set_text(val);
        }
    }

    /// Reset every value back to the empty placeholder.
    fn clear(&self) {
        for lbl in &self.values {
            lbl.set_text("\u{2014}");
        }
    }
}

/// RGB colour for a vertex by provenance origin. Native runtime emissions are
/// cyan; everything else is keyed off the origin label.
fn origin_color(v: &Vertex) -> (f64, f64, f64) {
    if v.origin.is_native() {
        return (0.0, 0.47, 0.72);
    }
    match v.origin.label().as_str() {
        "POST_RUN_DERIVED" => (0.49, 0.54, 0.63),
        "EXTERNAL_DERIVED" => (0.85, 0.46, 0.02),
        "BLOCKED" => (0.86, 0.15, 0.15),
        _ => (0.49, 0.54, 0.63),
    }
}

/// Append a rounded-rectangle path to the current cairo path.
fn rounded_rect(cr: &gtk::cairo::Context, x: f64, y: f64, w: f64, h: f64, r: f64) {
    use std::f64::consts::PI;
    let r = r.min(w / 2.0).min(h / 2.0).max(0.0);
    cr.new_sub_path();
    cr.arc(x + w - r, y + r, r, -0.5 * PI, 0.0);
    cr.arc(x + w - r, y + h - r, r, 0.0, 0.5 * PI);
    cr.arc(x + r, y + h - r, r, 0.5 * PI, PI);
    cr.arc(x + r, y + r, r, PI, 1.5 * PI);
    cr.close_path();
}

/// Stroke a small arrowhead pointing from `(x0,y0)` toward `(x1,y1)`.
fn draw_arrowhead(cr: &gtk::cairo::Context, x0: f64, y0: f64, x1: f64, y1: f64) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-6 {
        return;
    }
    let ux = dx / len;
    let uy = dy / len;
    // Pull the tip back a touch so it does not vanish inside the target node.
    let tip_x = x1 - ux * 4.0;
    let tip_y = y1 - uy * 4.0;
    let size = 9.0;
    let angle = 0.45_f64;
    let (ca, sa) = (angle.cos(), angle.sin());
    // Rotate the reverse-direction vector by +/- angle for the two barbs.
    let bx = -ux;
    let by = -uy;
    let lx = bx * ca - by * sa;
    let ly = bx * sa + by * ca;
    let rx = bx * ca + by * sa;
    let ry = -bx * sa + by * ca;
    cr.set_source_rgb(0.49, 0.54, 0.63);
    cr.set_line_width(1.5);
    cr.move_to(tip_x, tip_y);
    cr.line_to(tip_x + lx * size, tip_y + ly * size);
    cr.move_to(tip_x, tip_y);
    cr.line_to(tip_x + rx * size, tip_y + ry * size);
    let _ = cr.stroke();
}

/// Truncate a string to `max` characters with an ellipsis.
fn shorten(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let take = max.saturating_sub(1);
    let head: String = s.chars().take(take).collect();
    format!("{}\u{2026}", head)
}

/// Compute a layered left-to-right layout. Depth is the longest path from any
/// root (a vertex with no incoming edge); nodes at the same depth are stacked
/// vertically in discovery order.
fn layout_nodes(graph: &StateGraph) -> Vec<NodeBox> {
    const NODE_W: f64 = 180.0;
    const NODE_H: f64 = 54.0;

    let n = graph.vertices.len();
    if n == 0 {
        return Vec::new();
    }

    // First occurrence of each vertex id maps to its index.
    let mut id_to_index: HashMap<&str, usize> = HashMap::with_capacity(n);
    for (i, v) in graph.vertices.iter().enumerate() {
        id_to_index.entry(v.id.as_str()).or_insert(i);
    }

    // Adjacency + in-degree, restricted to edges whose endpoints are vertices.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indeg: Vec<usize> = vec![0; n];
    for e in &graph.edges {
        if let (Some(&s), Some(&t)) = (
            id_to_index.get(e.source.as_str()),
            id_to_index.get(e.target.as_str()),
        ) {
            adj[s].push(t);
            indeg[t] += 1;
        }
    }

    // Kahn's algorithm computing the longest path (depth) to each vertex.
    // Vertices trapped in a cycle never reach in-degree zero and keep depth 0,
    // which is a graceful fallback rather than an infinite loop.
    let mut depth = vec![0usize; n];
    let mut work = indeg.clone();
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (i, &d) in work.iter().enumerate() {
        if d == 0 {
            queue.push_back(i);
        }
    }
    while let Some(u) = queue.pop_front() {
        let du = depth[u];
        for &v in &adj[u] {
            if du + 1 > depth[v] {
                depth[v] = du + 1;
            }
            if work[v] > 0 {
                work[v] -= 1;
                if work[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
    }

    let mut count_at_depth: HashMap<usize, usize> = HashMap::new();
    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let d = depth[i];
        let slot = count_at_depth.entry(d).or_insert(0);
        let idx = *slot;
        *slot += 1;
        let x = 40.0 + (d as f64) * 220.0;
        let y = 40.0 + (idx as f64) * 90.0;
        nodes.push(NodeBox {
            vertex_index: i,
            x,
            y,
            w: NODE_W,
            h: NODE_H,
        });
    }
    nodes
}

/// Bounding extent (max x, max y) of a laid-out node set.
fn layout_extent(nodes: &[NodeBox]) -> (f64, f64) {
    let mut mx = 0.0_f64;
    let mut my = 0.0_f64;
    for nb in nodes {
        mx = mx.max(nb.x + nb.w);
        my = my.max(nb.y + nb.h);
    }
    (mx, my)
}

/// Read the state-graph JSONL from the first readable, non-empty candidate:
/// the active installed run directory. Returns an empty string when it is
/// absent, which the caller renders as an explicit empty state.
fn load_state_graph_text(ctx: &AppContext) -> String {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(dir) = ctx.run_dir() {
        candidates.push(dir.join(GRAPH_FILE));
    }
    for path in candidates {
        if let Ok(text) = std::fs::read_to_string(&path) {
            if !text.trim().is_empty() {
                return text;
            }
        }
    }
    String::new()
}

/// Build the DSTW-T4 state-graph visualization panel.
pub fn build_panel(ctx: &AppContext) -> gtk::Widget {
    // ---- shared panel-local state -----------------------------------------
    let graph_rc: Rc<RefCell<StateGraph>> = Rc::new(RefCell::new(StateGraph::default()));
    let nodes_rc: Rc<RefCell<Vec<NodeBox>>> = Rc::new(RefCell::new(Vec::new()));
    // (scale, off_x, off_y)
    let transform_rc: Rc<RefCell<(f64, f64, f64)>> = Rc::new(RefCell::new((1.0, 0.0, 0.0)));
    let drag_base_rc: Rc<RefCell<(f64, f64)>> = Rc::new(RefCell::new((0.0, 0.0)));
    let selected_rc: Rc<RefCell<Option<usize>>> = Rc::new(RefCell::new(None));
    let details = Rc::new(DetailsPane::new());

    // ---- drawing area ------------------------------------------------------
    let area = gtk::DrawingArea::new();
    area.set_hexpand(true);
    area.set_vexpand(true);
    area.set_content_width(700);
    area.set_content_height(500);

    {
        let graph_rc = graph_rc.clone();
        let nodes_rc = nodes_rc.clone();
        let transform_rc = transform_rc.clone();
        let selected_rc = selected_rc.clone();
        area.set_draw_func(move |_area, cr, width, height| {
            // Opaque white backdrop in device space.
            cr.set_source_rgb(1.0, 1.0, 1.0);
            cr.rectangle(0.0, 0.0, width as f64, height as f64);
            let _ = cr.fill();

            // Apply pan + zoom for the graph body.
            let (scale, off_x, off_y) = *transform_rc.borrow();
            cr.translate(off_x, off_y);
            cr.scale(scale, scale);

            let graph = graph_rc.borrow();
            let nodes = nodes_rc.borrow();

            // Centre lookup for edge endpoints.
            let mut centers: HashMap<&str, (f64, f64)> = HashMap::with_capacity(nodes.len());
            for nb in nodes.iter() {
                if let Some(v) = graph.vertices.get(nb.vertex_index) {
                    // First occurrence wins, matching id_to_index in layout_nodes
                    // (line 224). The durable T4 graph repeats a vertex id across
                    // lines, so last-write-wins here would route edges to a box
                    // the layout never positioned as that endpoint.
                    centers
                        .entry(v.id.as_str())
                        .or_insert((nb.x + nb.w / 2.0, nb.y + nb.h / 2.0));
                }
            }

            // Edges first, beneath the nodes.
            cr.set_line_width(1.5);
            for e in &graph.edges {
                if let (Some(&(sx, sy)), Some(&(tx, ty))) = (
                    centers.get(e.source.as_str()),
                    centers.get(e.target.as_str()),
                ) {
                    cr.set_source_rgb(0.49, 0.54, 0.63);
                    cr.move_to(sx, sy);
                    cr.line_to(tx, ty);
                    let _ = cr.stroke();
                    draw_arrowhead(cr, sx, sy, tx, ty);
                }
            }

            // Nodes.
            cr.set_font_size(12.0);
            let selected = *selected_rc.borrow();
            for (i, nb) in nodes.iter().enumerate() {
                let v = match graph.vertices.get(nb.vertex_index) {
                    Some(v) => v,
                    None => continue,
                };
                let (r, g, b) = origin_color(v);

                rounded_rect(cr, nb.x, nb.y, nb.w, nb.h, 8.0);
                cr.set_source_rgba(r, g, b, 0.16);
                let _ = cr.fill_preserve();
                cr.set_source_rgb(r, g, b);
                cr.set_line_width(if selected == Some(i) { 3.5 } else { 2.0 });
                let _ = cr.stroke();

                // Two text lines: shortened id, then state-object class.
                cr.set_source_rgb(0.06, 0.09, 0.14);
                cr.move_to(nb.x + 10.0, nb.y + 22.0);
                let _ = cr.show_text(&shorten(&v.id, 24));
                cr.move_to(nb.x + 10.0, nb.y + 41.0);
                let _ = cr.show_text(&shorten(&v.state_object_class, 26));

                // Decode-proof check glyph, top-right.
                if v.decode_proof_pass {
                    cr.set_source_rgb(0.06, 0.64, 0.43);
                    cr.set_line_width(2.0);
                    let gx = nb.x + nb.w - 18.0;
                    let gy = nb.y + 12.0;
                    cr.move_to(gx, gy);
                    cr.line_to(gx + 3.5, gy + 4.5);
                    cr.line_to(gx + 10.0, gy - 5.0);
                    let _ = cr.stroke();
                }
            }

            // Legend pinned near the graph origin, making native-vs-derived
            // unmistakable.
            let legend: [(&str, (f64, f64, f64)); 4] = [
                ("NATIVE", (0.0, 0.47, 0.72)),
                ("POST_RUN_DERIVED", (0.49, 0.54, 0.63)),
                ("EXTERNAL_DERIVED", (0.85, 0.46, 0.02)),
                ("BLOCKED", (0.86, 0.15, 0.15)),
            ];
            cr.set_font_size(11.0);
            let mut lx = 40.0_f64;
            let ly = 10.0_f64;
            for (name, (r, g, b)) in legend {
                rounded_rect(cr, lx, ly, 14.0, 14.0, 3.0);
                cr.set_source_rgba(r, g, b, 0.16);
                let _ = cr.fill_preserve();
                cr.set_source_rgb(r, g, b);
                cr.set_line_width(1.5);
                let _ = cr.stroke();

                cr.set_source_rgb(0.20, 0.24, 0.30);
                cr.move_to(lx + 20.0, ly + 11.5);
                let _ = cr.show_text(name);

                lx += 24.0 + (name.len() as f64) * 7.0 + 18.0;
            }
        });
    }

    // ---- left side: scrolled drawing + empty-state ------------------------
    let scrolled = gtk::ScrolledWindow::builder()
        .hexpand(true)
        .vexpand(true)
        .child(&area)
        .build();

    let empty_label = gtk::Label::new(Some(&format!(
        "{}\n\nNo state graph found in the active installed run directory.\n\nSelect a run that contains the DSTW-T4 binding object, then press Reload.",
        GRAPH_FILE
    )));
    empty_label.add_css_class("dim");
    empty_label.set_wrap(true);
    empty_label.set_justify(gtk::Justification::Center);
    empty_label.set_hexpand(true);
    empty_label.set_vexpand(true);
    empty_label.set_halign(gtk::Align::Center);
    empty_label.set_valign(gtk::Align::Center);
    empty_label.set_visible(false);

    let left_box = gtk::Box::new(gtk::Orientation::Vertical, 0);
    left_box.set_hexpand(true);
    left_box.set_vexpand(true);
    left_box.append(&scrolled);
    left_box.append(&empty_label);

    // ---- right side: details inspector ------------------------------------
    let details_scroll = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .min_content_width(340)
        .child(&details.container)
        .build();

    let paned = gtk::Paned::new(gtk::Orientation::Horizontal);
    paned.set_hexpand(true);
    paned.set_vexpand(true);
    paned.set_start_child(Some(&left_box));
    paned.set_end_child(Some(&details_scroll));
    paned.set_position(760);
    paned.set_shrink_start_child(false);
    paned.set_shrink_end_child(false);

    // ---- header bar --------------------------------------------------------
    let header = gtk::Box::new(gtk::Orientation::Horizontal, 12);
    header.add_css_class("card");
    header.set_margin_top(10);
    header.set_margin_bottom(10);
    header.set_margin_start(12);
    header.set_margin_end(12);

    let title = gtk::Label::new(Some("Scientific State Graph"));
    title.add_css_class("prismos-title");
    title.set_xalign(0.0);

    let v_label = gtk::Label::new(Some("vertices 0"));
    v_label.add_css_class("mono");
    let e_label = gtk::Label::new(Some("edges 0"));
    e_label.add_css_class("mono");
    let n_label = gtk::Label::new(Some("native 0/0"));
    n_label.add_css_class("mono");

    let pill = gtk::Label::new(Some("loading"));
    pill.add_css_class("status-pill");
    pill.add_css_class("info");

    let spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    spacer.set_hexpand(true);

    let reload_btn = gtk::Button::with_label("Reload");
    reload_btn.add_css_class("accent-button");

    header.append(&title);
    header.append(&v_label);
    header.append(&e_label);
    header.append(&n_label);
    header.append(&pill);
    header.append(&spacer);
    header.append(&reload_btn);

    // ---- rebuild (initial load + Reload) ----------------------------------
    let rebuild: Rc<dyn Fn()> = Rc::new({
        let ctx = ctx.clone();
        let area = area.clone();
        let scrolled = scrolled.clone();
        let empty_label = empty_label.clone();
        let v_label = v_label.clone();
        let e_label = e_label.clone();
        let n_label = n_label.clone();
        let pill = pill.clone();
        let graph_rc = graph_rc.clone();
        let nodes_rc = nodes_rc.clone();
        let transform_rc = transform_rc.clone();
        let selected_rc = selected_rc.clone();
        let details = details.clone();
        move || {
            let text = load_state_graph_text(&ctx);
            let graph = StateGraph::parse_jsonl(&text);
            let nodes = layout_nodes(&graph);
            let (mx, my) = layout_extent(&nodes);

            area.set_content_width((mx + 220.0).max(700.0) as i32);
            area.set_content_height((my + 120.0).max(500.0) as i32);

            v_label.set_text(&format!("vertices {}", graph.vertices.len()));
            e_label.set_text(&format!("edges {}", graph.edges.len()));
            n_label.set_text(&format!(
                "native {}/{}",
                graph.native_vertex_count(),
                graph.vertices.len()
            ));

            pill.remove_css_class("info");
            pill.remove_css_class("pass");
            pill.remove_css_class("blocked");
            pill.remove_css_class("warn");
            if graph.vertices.is_empty() {
                pill.set_text("no state graph");
                pill.add_css_class("warn");
            } else if graph.all_decode_proofs_pass() {
                pill.set_text("decode proofs PASS");
                pill.add_css_class("pass");
            } else {
                pill.set_text("decode proof gaps");
                pill.add_css_class("blocked");
            }

            let empty = graph.vertices.is_empty();
            scrolled.set_visible(!empty);
            empty_label.set_visible(empty);

            *graph_rc.borrow_mut() = graph;
            *nodes_rc.borrow_mut() = nodes;
            *selected_rc.borrow_mut() = None;
            *transform_rc.borrow_mut() = (1.0, 0.0, 0.0);
            details.clear();
            area.queue_draw();
        }
    });
    rebuild();

    {
        let rb = rebuild.clone();
        reload_btn.connect_clicked(move |_| rb());
    }

    // ---- interaction: click to inspect ------------------------------------
    let click = gtk::GestureClick::new();
    {
        let graph_rc = graph_rc.clone();
        let nodes_rc = nodes_rc.clone();
        let transform_rc = transform_rc.clone();
        let selected_rc = selected_rc.clone();
        let details = details.clone();
        let area = area.clone();
        click.connect_pressed(move |_g, _n, sx, sy| {
            let (scale, off_x, off_y) = *transform_rc.borrow();
            if scale.abs() < 1e-9 {
                return;
            }
            let gx = (sx - off_x) / scale;
            let gy = (sy - off_y) / scale;

            let nodes = nodes_rc.borrow();
            let graph = graph_rc.borrow();
            let mut hit: Option<usize> = None;
            for (i, nb) in nodes.iter().enumerate() {
                if gx >= nb.x && gx <= nb.x + nb.w && gy >= nb.y && gy <= nb.y + nb.h {
                    hit = Some(i);
                }
            }
            if let Some(i) = hit {
                if let Some(v) = graph.vertices.get(nodes[i].vertex_index) {
                    details.show(v);
                }
                *selected_rc.borrow_mut() = Some(i);
                area.queue_draw();
            }
        });
    }
    area.add_controller(click);

    // ---- interaction: scroll to zoom -------------------------------------
    let scroll = gtk::EventControllerScroll::new(gtk::EventControllerScrollFlags::VERTICAL);
    {
        let transform_rc = transform_rc.clone();
        let area = area.clone();
        scroll.connect_scroll(move |_c, _dx, dy| {
            {
                let mut t = transform_rc.borrow_mut();
                t.0 = (t.0 * (1.0 - dy * 0.1)).clamp(0.3, 3.0);
            }
            area.queue_draw();
            glib::Propagation::Stop
        });
    }
    area.add_controller(scroll);

    // ---- interaction: drag to pan ----------------------------------------
    let drag = gtk::GestureDrag::new();
    {
        let transform_rc = transform_rc.clone();
        let drag_base_rc = drag_base_rc.clone();
        drag.connect_drag_begin(move |_g, _x, _y| {
            let t = transform_rc.borrow();
            *drag_base_rc.borrow_mut() = (t.1, t.2);
        });
    }
    {
        let transform_rc = transform_rc.clone();
        let drag_base_rc = drag_base_rc.clone();
        let area = area.clone();
        drag.connect_drag_update(move |_g, ox, oy| {
            let base = *drag_base_rc.borrow();
            {
                let mut t = transform_rc.borrow_mut();
                t.1 = base.0 + ox;
                t.2 = base.1 + oy;
            }
            area.queue_draw();
        });
    }
    area.add_controller(drag);

    // ---- assemble root -----------------------------------------------------
    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    root.add_css_class("prismos-root");
    root.set_hexpand(true);
    root.set_vexpand(true);
    root.append(&header);
    root.append(&paned);
    root.upcast()
}
