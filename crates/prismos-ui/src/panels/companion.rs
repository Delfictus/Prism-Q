//! Prism CoPilot — the conversational agent panel.
//!
//! gpt-oss front-end that calls the DSTW tools (PrismAI physics, binding sites,
//! affinity scoring, analogs, pocket audit, design) on demand. Colored chat
//! bubbles (no "you/companion" labels), a multi-line wrapping input, file
//! attach + answer download, and clickable follow-up chips.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use adw::prelude::*;

use crate::adapters::PrismCommand;
use crate::app::AppContext;
use crate::jobs::JobEvent;

struct CoPilot {
    ctx: AppContext,
    messages: gtk::Box,
    scroller: gtk::ScrolledWindow,
    input: gtk::TextView,
    send: gtk::Button,
    attachment: RefCell<Option<PathBuf>>,
    attach_label: gtk::Label,
    last_answer: RefCell<String>,
}

/// Build the Prism CoPilot panel.
pub fn build(ctx: &AppContext) -> gtk::Widget {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 10);
    root.add_css_class("prismos-root");
    root.set_margin_top(14);
    root.set_margin_bottom(14);
    root.set_margin_start(14);
    root.set_margin_end(14);

    // header: title + New conversation + Download
    let header = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let titles = gtk::Box::new(gtk::Orientation::Vertical, 1);
    titles.set_hexpand(true);
    let title = gtk::Label::new(Some("Prism CoPilot"));
    title.add_css_class("prismos-title");
    title.set_halign(gtk::Align::Start);
    let subtitle = gtk::Label::new(Some(
        "Ask about targets, pockets, molecules, or designs. It calls the DSTW tools \
         (PrismAI physics, binding sites, affinity scoring) when it needs real data.",
    ));
    subtitle.add_css_class("subtitle");
    subtitle.set_halign(gtk::Align::Start);
    subtitle.set_wrap(true);
    titles.append(&title);
    titles.append(&subtitle);
    header.append(&titles);
    let download_btn = gtk::Button::from_icon_name("document-save-symbolic");
    download_btn.set_tooltip_text(Some("Download last answer"));
    download_btn.set_valign(gtk::Align::Center);
    let reset_btn = gtk::Button::with_label("New conversation");
    reset_btn.set_valign(gtk::Align::Center);
    header.append(&download_btn);
    header.append(&reset_btn);
    root.append(&header);

    // messages list
    let messages = gtk::Box::new(gtk::Orientation::Vertical, 8);
    messages.set_margin_top(6);
    messages.set_margin_bottom(6);
    let scroller = gtk::ScrolledWindow::builder()
        .hexpand(true)
        .vexpand(true)
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&messages)
        .build();
    scroller.add_css_class("card");
    root.append(&scroller);

    // attachment indicator
    let attach_label = gtk::Label::new(None);
    attach_label.add_css_class("copilot-attach");
    attach_label.set_halign(gtk::Align::Start);
    root.append(&attach_label);

    // input row: attach + multi-line wrapping input + send
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let attach_btn = gtk::Button::from_icon_name("mail-attachment-symbolic");
    attach_btn.set_tooltip_text(Some("Attach a file (PDB, trajectory, …)"));
    attach_btn.set_valign(gtk::Align::End);
    let input = gtk::TextView::new();
    input.set_wrap_mode(gtk::WrapMode::WordChar);
    input.set_accepts_tab(false);
    input.set_top_margin(6);
    input.set_bottom_margin(6);
    input.set_left_margin(8);
    input.set_right_margin(8);
    let input_scroll = gtk::ScrolledWindow::builder()
        .hexpand(true)
        .min_content_height(44)
        .max_content_height(130)
        .propagate_natural_height(true)
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&input)
        .build();
    input_scroll.add_css_class("card");
    input_scroll.add_css_class("copilot-input-card");
    let send = gtk::Button::with_label("Send");
    send.add_css_class("accent-button");
    send.set_valign(gtk::Align::End);
    row.append(&attach_btn);
    row.append(&input_scroll);
    row.append(&send);
    root.append(&row);

    let ui = Rc::new(CoPilot {
        ctx: ctx.clone(),
        messages: messages.clone(),
        scroller: scroller.clone(),
        input: input.clone(),
        send: send.clone(),
        attachment: RefCell::new(None),
        attach_label: attach_label.clone(),
        last_answer: RefCell::new(String::new()),
    });

    add_bubble(
        &messages,
        "Prism CoPilot ready. Ask a question, or attach a file to analyze.",
        false,
        None,
    );

    // send button
    {
        let ui = ui.clone();
        send.connect_clicked(move |_| send_current(&ui));
    }
    // Enter to send, Shift+Enter for newline
    {
        let ui = ui.clone();
        let key = gtk::EventControllerKey::new();
        key.connect_key_pressed(move |_, keyval, _code, state| {
            if keyval == gtk::gdk::Key::Return
                && !state.contains(gtk::gdk::ModifierType::SHIFT_MASK)
            {
                send_current(&ui);
                glib::Propagation::Stop
            } else {
                glib::Propagation::Proceed
            }
        });
        input.add_controller(key);
    }
    // attach
    {
        let ui = ui.clone();
        attach_btn.connect_clicked(move |_| {
            let dialog = gtk::FileDialog::new();
            dialog.set_title("Attach a file for Prism CoPilot");
            let ui = ui.clone();
            let window = ui.ctx.window.clone();
            dialog.open(Some(&window), gtk::gio::Cancellable::NONE, move |result| {
                if let Ok(file) = result {
                    if let Some(path) = file.path() {
                        let name = path
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default();
                        ui.attach_label.set_text(&format!(
                            "\u{1F4CE} attached: {name}  (sent with next message)"
                        ));
                        *ui.attachment.borrow_mut() = Some(path);
                    }
                }
            });
        });
    }
    // download last answer
    {
        let ui = ui.clone();
        download_btn.connect_clicked(move |_| {
            let text = ui.last_answer.borrow().clone();
            if text.is_empty() {
                ui.ctx.toast("No answer to download yet");
                return;
            }
            let dialog = gtk::FileDialog::new();
            dialog.set_title("Save answer");
            dialog.set_initial_name(Some("prism_copilot_answer.md"));
            let ctx = ui.ctx.clone();
            let window = ui.ctx.window.clone();
            dialog.save(Some(&window), gtk::gio::Cancellable::NONE, move |result| {
                if let Ok(file) = result {
                    if let Some(path) = file.path() {
                        match std::fs::write(&path, &text) {
                            Ok(_) => ctx.toast(&format!("Saved {}", path.display())),
                            Err(e) => ctx.toast(&format!("Save failed: {e}")),
                        }
                    }
                }
            });
        });
    }
    // new conversation
    {
        let ui = ui.clone();
        reset_btn.connect_clicked(move |_| {
            while let Some(child) = ui.messages.first_child() {
                ui.messages.remove(&child);
            }
            add_bubble(&ui.messages, "New conversation started.", false, None);
            *ui.attachment.borrow_mut() = None;
            ui.attach_label.set_text("");
            let cmd = PrismCommand::companion_reset();
            ui.ctx.jobs.spawn(cmd, &ui.ctx.resolver, move |_| {});
        });
    }

    root.upcast()
}

fn send_current(ui: &Rc<CoPilot>) {
    let buf = ui.input.buffer();
    let text = buf
        .text(&buf.start_iter(), &buf.end_iter(), false)
        .to_string();
    send_text(ui, text.trim().to_string());
}

fn send_text(ui: &Rc<CoPilot>, text: String) {
    let attach = ui.attachment.borrow().clone();
    if text.is_empty() && attach.is_none() {
        return;
    }
    ui.send.set_sensitive(false);
    ui.input.buffer().set_text("");

    let shown = match &attach {
        Some(p) => format!(
            "{text}\n\u{1F4CE} {}",
            p.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default()
        ),
        None => text.clone(),
    };
    add_bubble(&ui.messages, shown.trim(), true, None);
    *ui.attachment.borrow_mut() = None;
    ui.attach_label.set_text("");

    // thinking placeholder
    let thinking = make_bubble("\u{2026}", false);
    ui.messages.append(&thinking);
    scroll_to_bottom(&ui.scroller);

    let cmd = PrismCommand::copilot_chat(&text, attach.as_deref());
    let acc = Rc::new(RefCell::new(String::new()));
    let ui2 = ui.clone();
    let thinking2 = thinking.clone();
    ui.ctx
        .jobs
        .spawn(cmd, &ui.ctx.resolver, move |ev| match ev {
            JobEvent::Stdout(line) => {
                acc.borrow_mut().push_str(&line);
                acc.borrow_mut().push('\n');
            }
            JobEvent::Stderr(_) => {}
            JobEvent::Done(outcome) => {
                ui2.messages.remove(&thinking2);
                let (answer, tools, followups) = parse_output(&acc.borrow());
                *ui2.last_answer.borrow_mut() = answer.clone();
                add_bubble(
                    &ui2.messages,
                    if answer.is_empty() {
                        "(no answer)"
                    } else {
                        &answer
                    },
                    false,
                    tools.as_deref(),
                );
                if !followups.is_empty() {
                    add_chips(&ui2, &followups);
                }
                scroll_to_bottom(&ui2.scroller);
                ui2.send.set_sensitive(true);
                if !outcome.success {
                    ui2.ctx.toast("Prism CoPilot turn failed");
                }
            }
        });
}

/// Split raw prism-companion stdout into (answer, tools footnote, followups).
fn parse_output(raw: &str) -> (String, Option<String>, Vec<String>) {
    let mut answer = Vec::new();
    let mut tools = None;
    let mut followups = Vec::new();
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("@@TOOLS@@ ") {
            tools = Some(format!("via {}", rest.trim()));
        } else if let Some(rest) = line.strip_prefix("@@FOLLOWUPS@@ ") {
            if let Ok(v) = serde_json::from_str::<Vec<String>>(rest.trim()) {
                followups = v;
            }
        } else {
            answer.push(line);
        }
    }
    (answer.join("\n").trim().to_string(), tools, followups)
}

fn make_bubble(text: &str, is_user: bool) -> gtk::Box {
    let bubble = gtk::Box::new(gtk::Orientation::Vertical, 4);
    bubble.add_css_class("copilot-bubble");
    bubble.add_css_class(if is_user {
        "copilot-user"
    } else {
        "copilot-ai"
    });
    bubble.set_halign(if is_user {
        gtk::Align::End
    } else {
        gtk::Align::Start
    });
    let label = gtk::Label::new(Some(text));
    label.set_wrap(true);
    label.set_wrap_mode(gtk::pango::WrapMode::WordChar);
    label.set_xalign(0.0);
    label.set_selectable(true);
    label.set_max_width_chars(64);
    bubble.append(&label);
    bubble
}

fn add_bubble(messages: &gtk::Box, text: &str, is_user: bool, tools: Option<&str>) {
    let bubble = make_bubble(text, is_user);
    if let Some(t) = tools {
        let foot = gtk::Label::new(Some(t));
        foot.add_css_class("copilot-tools");
        foot.set_xalign(0.0);
        foot.set_wrap(true);
        bubble.append(&foot);
    }
    messages.append(&bubble);
}

fn add_chips(ui: &Rc<CoPilot>, followups: &[String]) {
    let flow = gtk::FlowBox::new();
    flow.set_selection_mode(gtk::SelectionMode::None);
    flow.set_halign(gtk::Align::Start);
    flow.set_max_children_per_line(3);
    flow.set_column_spacing(6);
    flow.set_row_spacing(6);
    for q in followups {
        let chip = gtk::Button::with_label(q);
        chip.add_css_class("copilot-chip");
        let ui = ui.clone();
        let text = q.clone();
        chip.connect_clicked(move |_| send_text(&ui, text.clone()));
        flow.insert(&chip, -1);
    }
    ui.messages.append(&flow);
}

fn scroll_to_bottom(scroller: &gtk::ScrolledWindow) {
    let adj = scroller.vadjustment();
    glib::idle_add_local_once(move || adj.set_value(adj.upper()));
}
