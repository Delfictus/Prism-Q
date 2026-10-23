//! # PRISM-Zero Flight Recorder Dashboard
//!
//! Real-time TUI monitoring for physics engine telemetry data.
//! Displays energy convergence, temperature control, and system performance
//! without impacting the simulation hot path.
//!
//! ## Usage
//! ```bash
//! cargo run --bin prism-monitor --features telemetry
//! ```
//!
//! ## Dashboard Layout
//! ```text
//! ┌─ PRISM-Zero Flight Recorder ─────────────────────────┐
//! │ Step: 1,234,567  │  Energy: -1,234.56 kcal/mol     │
//! │ Time: 12m 34s    │  Temp: 300.15 K                 │
//! │ Rate: 2.1 ms/cyc │  Accept: 85.3%                  │
//! ├──────────────────┼──────────────────────────────────┤
//! │     Energy       │          Temperature             │
//! │ ─────────────────│─────────────────────────────────│
//! │   -1234.56 ┤     │    300.15 ┤                     │
//! │   -1235.78 ┤     │    299.87 ┤                     │
//! │   -1236.12 ┤     │    300.43 ┤                     │
//! │            ┴─────│           ┴─────────────────────│
//! ├──────────────────┼──────────────────────────────────┤
//! │   Acceptance     │          Gradient                │
//! │ ─────────────────│─────────────────────────────────│
//! │     85.3% ┤      │     0.012 ┤                     │
//! │     84.7% ┤      │     0.008 ┤                     │
//! │     86.1% ┤      │     0.015 ┤                     │
//! │           ┴──────│           ┴─────────────────────│
//! └──────────────────────────────────────────────────────┘
//! ```

use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use prism_core::telemetry::{self, TelemetryFrame};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph},
};
use std::{
    collections::VecDeque,
    io::{self, Stdout},
    time::{Duration, Instant},
};

/// Maximum history length for chart display
const MAX_HISTORY: usize = 1000;

/// Dashboard refresh rate (60 FPS for smooth visualization)
const REFRESH_RATE: Duration = Duration::from_millis(16);

/// Telemetry history storage for visualization
#[derive(Debug, Clone)]
struct TelemetryHistory {
    energy: VecDeque<(f64, f64)>,
    temperature: VecDeque<(f64, f64)>,
    acceptance: VecDeque<(f64, f64)>,
    gradient: VecDeque<(f64, f64)>,
    start_time: Instant,
}

impl TelemetryHistory {
    fn new() -> Self {
        Self {
            energy: VecDeque::with_capacity(MAX_HISTORY),
            temperature: VecDeque::with_capacity(MAX_HISTORY),
            acceptance: VecDeque::with_capacity(MAX_HISTORY),
            gradient: VecDeque::with_capacity(MAX_HISTORY),
            start_time: Instant::now(),
        }
    }

    fn add_frame(&mut self, frame: &TelemetryFrame) {
        let time = frame.timestamp_ns as f64 / 1e9; // Convert to seconds

        // Add new data points
        self.energy.push_back((time, frame.energy as f64));
        self.temperature.push_back((time, frame.temperature as f64));
        self.acceptance
            .push_back((time, frame.acceptance_rate as f64 * 100.0));
        self.gradient.push_back((time, frame.gradient_norm as f64));

        // Maintain maximum history size
        while self.energy.len() > MAX_HISTORY {
            self.energy.pop_front();
        }
        while self.temperature.len() > MAX_HISTORY {
            self.temperature.pop_front();
        }
        while self.acceptance.len() > MAX_HISTORY {
            self.acceptance.pop_front();
        }
        while self.gradient.len() > MAX_HISTORY {
            self.gradient.pop_front();
        }
    }

    fn latest_frame(&self) -> Option<TelemetryFrame> {
        if let (Some(&(time, energy)), Some(&(_, temp)), Some(&(_, acc)), Some(&(_, grad))) = (
            self.energy.back(),
            self.temperature.back(),
            self.acceptance.back(),
            self.gradient.back(),
        ) {
            Some(TelemetryFrame {
                step: 0, // We'll track this separately
                timestamp_ns: (time * 1e9) as u64,
                energy: energy as f32,
                temperature: temp as f32,
                acceptance_rate: (acc / 100.0) as f32,
                gradient_norm: grad as f32,
            })
        } else {
            None
        }
    }
}

/// Dashboard state and statistics
#[derive(Debug)]
struct DashboardState {
    history: TelemetryHistory,
    current_step: u64,
    total_frames: u64,
    start_time: Instant,
    paused: bool,
    show_help: bool,
}

impl DashboardState {
    fn new() -> Self {
        Self {
            history: TelemetryHistory::new(),
            current_step: 0,
            total_frames: 0,
            start_time: Instant::now(),
            paused: false,
            show_help: false,
        }
    }

    fn update_telemetry(&mut self, frames: &[TelemetryFrame]) {
        for frame in frames {
            self.history.add_frame(frame);
            self.current_step = frame.step;
            self.total_frames += 1;
        }
    }

    fn runtime_duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    fn average_cycle_time(&self) -> f64 {
        if self.current_step > 0 {
            self.runtime_duration().as_millis() as f64 / self.current_step as f64
        } else {
            0.0
        }
    }
}

type Terminal = ratatui::Terminal<CrosstermBackend<Stdout>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize telemetry system
    telemetry::init_telemetry();

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Dashboard state
    let mut state = DashboardState::new();

    // Main dashboard loop
    // The monitor is a passive observer - it displays only real telemetry data
    // No demo simulation: Zero-Mock Protocol enforcement
    let result = run_dashboard(&mut terminal, &mut state);

    // Cleanup terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn run_dashboard(
    terminal: &mut Terminal,
    state: &mut DashboardState,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        // Handle input events
        if event::poll(REFRESH_RATE)? {
            if let Event::Key(key) = event::read()? {
                match handle_key_event(key, state) {
                    KeyAction::Quit => break,
                    KeyAction::Continue => {}
                }
            }
        }

        // Update telemetry data (if not paused)
        if !state.paused {
            let frames = telemetry::drain_frames();
            if !frames.is_empty() {
                state.update_telemetry(&frames);
            }
        }

        // Render dashboard
        terminal.draw(|f| render_dashboard(f, state))?;
    }

    Ok(())
}

#[derive(Debug)]
enum KeyAction {
    Quit,
    Continue,
}

fn handle_key_event(key: KeyEvent, state: &mut DashboardState) -> KeyAction {
    if key.kind != KeyEventKind::Press {
        return KeyAction::Continue;
    }

    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => KeyAction::Quit,
        KeyCode::Char(' ') => {
            state.paused = !state.paused;
            KeyAction::Continue
        }
        KeyCode::Char('h') => {
            state.show_help = !state.show_help;
            KeyAction::Continue
        }
        KeyCode::Char('r') => {
            // Reset statistics
            state.start_time = Instant::now();
            state.current_step = 0;
            state.total_frames = 0;
            KeyAction::Continue
        }
        _ => KeyAction::Continue,
    }
}

fn render_dashboard(f: &mut ratatui::Frame, state: &DashboardState) {
    let size = f.size();

    if state.show_help {
        render_help_screen(f, size);
        return;
    }

    // Main layout: header + charts
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(0)])
        .split(size);

    // Render header with current metrics
    render_header(f, chunks[0], state);

    // Chart layout: 2x2 grid
    let chart_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    let top_charts = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chart_chunks[0]);

    let bottom_charts = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chart_chunks[1]);

    // Render charts
    render_energy_chart(f, top_charts[0], state);
    render_temperature_chart(f, top_charts[1], state);
    render_acceptance_chart(f, bottom_charts[0], state);
    render_gradient_chart(f, bottom_charts[1], state);
}

fn render_header(f: &mut ratatui::Frame, area: Rect, state: &DashboardState) {
    let latest = state.history.latest_frame();
    let runtime = state.runtime_duration();
    let cycle_time = state.average_cycle_time();

    let status = if state.paused { "PAUSED" } else { "RECORDING" };
    let status_style = if state.paused {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::Green)
    };

    let header_text = if let Some(frame) = latest {
        vec![
            Line::from(vec![
                Span::styled("🛩️  PRISM-Zero Flight Recorder ", Style::default().bold()),
                Span::styled(status, status_style.add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::raw(format!("Step: {:>10} ", state.current_step)),
                Span::raw("│ "),
                Span::raw(format!("Energy: {:>10.2} kcal/mol ", frame.energy)),
                Span::raw("│ "),
                Span::raw(format!("Runtime: {:>6.0}s", runtime.as_secs())),
            ]),
            Line::from(vec![
                Span::raw(format!("Temp: {:>10.2} K ", frame.temperature)),
                Span::raw("│ "),
                Span::raw(format!("Accept: {:>8.1}% ", frame.acceptance_rate * 100.0)),
                Span::raw("│ "),
                Span::raw(format!("Cycle: {:>6.1} ms", cycle_time)),
            ]),
        ]
    } else {
        vec![
            Line::from("🛩️  PRISM-Zero Flight Recorder - Waiting for telemetry data..."),
            Line::from("Connect a PRISM physics engine to begin monitoring"),
            Line::from("Press 'h' for help, 'q' to quit"),
        ]
    };

    let paragraph = Paragraph::new(header_text)
        .block(Block::default().borders(Borders::ALL).title("Status"))
        .alignment(Alignment::Left);

    f.render_widget(paragraph, area);
}

fn render_energy_chart(f: &mut ratatui::Frame, area: Rect, state: &DashboardState) {
    let data: Vec<(f64, f64)> = state.history.energy.iter().cloned().collect();

    if data.is_empty() {
        render_empty_chart(f, area, "Energy (kcal/mol)");
        return;
    }

    let min_energy = data.iter().map(|(_, e)| *e).fold(f64::INFINITY, f64::min);
    let max_energy = data
        .iter()
        .map(|(_, e)| *e)
        .fold(f64::NEG_INFINITY, f64::max);

    let dataset = Dataset::default()
        .name("Hamiltonian")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Cyan))
        .data(&data);

    let chart = Chart::new(vec![dataset])
        .block(
            Block::default()
                .title("Energy Convergence")
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("Time (s)")
                .style(Style::default().fg(Color::Gray)),
        )
        .y_axis(
            Axis::default()
                .title("Energy")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_energy - 1.0, max_energy + 1.0]),
        );

    f.render_widget(chart, area);
}

fn render_temperature_chart(f: &mut ratatui::Frame, area: Rect, state: &DashboardState) {
    let data: Vec<(f64, f64)> = state.history.temperature.iter().cloned().collect();

    if data.is_empty() {
        render_empty_chart(f, area, "Temperature (K)");
        return;
    }

    let dataset = Dataset::default()
        .name("Thermostat")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Red))
        .data(&data);

    let chart = Chart::new(vec![dataset])
        .block(
            Block::default()
                .title("Temperature Control")
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("Time (s)")
                .style(Style::default().fg(Color::Gray)),
        )
        .y_axis(
            Axis::default()
                .title("Temperature")
                .style(Style::default().fg(Color::Gray))
                .bounds([250.0, 350.0]),
        );

    f.render_widget(chart, area);
}

fn render_acceptance_chart(f: &mut ratatui::Frame, area: Rect, state: &DashboardState) {
    let data: Vec<(f64, f64)> = state.history.acceptance.iter().cloned().collect();

    if data.is_empty() {
        render_empty_chart(f, area, "Acceptance Rate (%)");
        return;
    }

    let dataset = Dataset::default()
        .name("MC Accept")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Green))
        .data(&data);

    let chart = Chart::new(vec![dataset])
        .block(
            Block::default()
                .title("Monte Carlo Acceptance")
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("Time (s)")
                .style(Style::default().fg(Color::Gray)),
        )
        .y_axis(
            Axis::default()
                .title("Acceptance %")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, 100.0]),
        );

    f.render_widget(chart, area);
}

fn render_gradient_chart(f: &mut ratatui::Frame, area: Rect, state: &DashboardState) {
    let data: Vec<(f64, f64)> = state.history.gradient.iter().cloned().collect();

    if data.is_empty() {
        render_empty_chart(f, area, "Gradient Norm");
        return;
    }

    let max_grad = data.iter().map(|(_, g)| *g).fold(0.0, f64::max);

    let dataset = Dataset::default()
        .name("‖∇H‖")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Magenta))
        .data(&data);

    let chart = Chart::new(vec![dataset])
        .block(
            Block::default()
                .title("Gradient Convergence")
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("Time (s)")
                .style(Style::default().fg(Color::Gray)),
        )
        .y_axis(
            Axis::default()
                .title("‖∇H‖")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, max_grad * 1.1]),
        );

    f.render_widget(chart, area);
}

fn render_empty_chart(f: &mut ratatui::Frame, area: Rect, title: &str) {
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::DarkGray));

    let paragraph = Paragraph::new("No data available")
        .block(block)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));

    f.render_widget(paragraph, area);
}

fn render_help_screen(f: &mut ratatui::Frame, area: Rect) {
    let help_text = vec![
        Line::from("🛩️  PRISM-Zero Flight Recorder - Help"),
        Line::from(""),
        Line::from("Keyboard Controls:"),
        Line::from("  q / ESC  - Quit dashboard"),
        Line::from("  SPACE    - Pause/Resume recording"),
        Line::from("  h        - Toggle this help screen"),
        Line::from("  r        - Reset statistics"),
        Line::from(""),
        Line::from("Charts:"),
        Line::from("  Top Left     - Energy Convergence (Hamiltonian)"),
        Line::from("  Top Right    - Temperature Control (Thermostat)"),
        Line::from("  Bottom Left  - Monte Carlo Acceptance Rate"),
        Line::from("  Bottom Right - Gradient Convergence"),
        Line::from(""),
        Line::from("Architecture:"),
        Line::from("  • Lock-free ring buffer for <5ns recording"),
        Line::from("  • 60 FPS dashboard refresh"),
        Line::from("  • Zero impact on physics engine performance"),
        Line::from(""),
        Line::from("Press 'h' to return to dashboard"),
    ];

    let paragraph = Paragraph::new(help_text)
        .block(Block::default().title("Help").borders(Borders::ALL))
        .alignment(Alignment::Left)
        .style(Style::default().fg(Color::White));

    f.render_widget(paragraph, area);
}
