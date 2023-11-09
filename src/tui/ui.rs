//! Defines the Terminal User Interface (TUI) for NPcore

use eyre::Result;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    Frame, Terminal,
};
use std::{
    io::stdout,
    time::{Duration, Instant},
};
use tokio::sync::mpsc::UnboundedReceiver;

use super::{
    inputs::{events::Events, InputEvent},
    state::CycleHistory,
    App, AppReturn,
};

pub enum Comm {
    NPCycle(NPCycle),
    Message(String),
    Stop(bool),
}

use crate::prelude::{output::NPCycle, settings::run::Data};
use crate::tui::components::*;

pub fn start_ui(mut rx: UnboundedReceiver<Comm>, settings: Data) -> Result<()> {
    let stdout = stdout();
    crossterm::terminal::enable_raw_mode()?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::new();
    let mut cycle_history = CycleHistory::new();

    terminal.clear()?;

    // User event handler
    let tick_rate = Duration::from_millis(200);
    let mut events = Events::new(tick_rate);

    let start_time = Instant::now();
    let mut elapsed_time = Duration::from_secs(0);
    // Main UI loop
    loop {
        let _ = match rx.try_recv() {
            Ok(comm) => match comm {
                Comm::NPCycle(cycle) => {
                    app.state = cycle.clone();
                    cycle_history.add_cycle(cycle);
                }
                Comm::Message(_msg) => {}
                Comm::Stop(stop) => {
                    if stop {
                        break; //TODO: Replace with graceful exit from TUI
                    }
                }
            },
            Err(_) => {}
        };

        // Update elapsed time
        let now = Instant::now();
        elapsed_time = now.duration_since(start_time);

        // Draw the terminal
        terminal
            .draw(|rect| draw(rect, &app, &cycle_history, elapsed_time, &settings))
            .unwrap();

        // Handle inputs
        let result = match events.recv() {
            Some(InputEvent::Input(key)) => app.do_action(key),
            None => AppReturn::Continue,
        };
        // Check if we should exit
        if result == AppReturn::Exit {
            break;
        }
    }

    terminal
        .draw(|rect| draw(rect, &app, &cycle_history, elapsed_time, &settings))
        .unwrap();
    println!();
    terminal.show_cursor()?;
    crossterm::terminal::disable_raw_mode()?;

    Ok(())
}

pub fn draw(
    rect: &mut Frame,
    app: &App,
    cycle_history: &CycleHistory,
    elapsed_time: Duration,
    settings: &Data,
) {
    let size = rect.size();

    // Vertical layout (overall)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Min(5),
            ]
            .as_ref(),
        )
        .split(size);

    // Title in first chunk (top)
    let title = draw_title();
    rect.render_widget(title, chunks[0]);

    // Horizontal layout for three chunks (middle)
    let body_chunk = chunks[1];
    let body_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(
            [
                Constraint::Percentage(40),
                Constraint::Percentage(40),
                Constraint::Percentage(20),
            ]
            .as_ref(),
        )
        .split(body_chunk);

    // First chunk
    let status = draw_status(app, elapsed_time);
    rect.render_widget(status, body_layout[0]);

    // Second chunk
    let options = draw_options(settings);
    rect.render_widget(options, body_layout[1]);

    // Third chunk
    let commands = draw_commands(app);
    rect.render_widget(commands, body_layout[2]);

    // Bottom chunk (tabs)
    let bottom_chunk = chunks[2];
    let tab_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(10), Constraint::Percentage(90)].as_ref())
        .split(bottom_chunk);

    let tabs = draw_tabs(&app);
    rect.render_widget(tabs, tab_layout[0]);

    // Plot
    // Prepare the data
    let data: Vec<(f64, f64)> = cycle_history
        .cycles
        .iter()
        .enumerate()
        .map(|(x, entry)| (x as f64, entry.objf))
        .collect();

    let start_index = (data.len() as f64 * 0.1) as usize;

    // Calculate data points and remove infinities
    let mut norm_data: Vec<(f64, f64)> = data
        .iter()
        .filter(|&(_, y)| !y.is_infinite())
        .skip(start_index)
        .map(|&(x, y)| (x, y))
        .collect();

    // Tab content
    let inner_height = tab_layout[1].height;
    match app.tab_index {
        0 => {
            let logs = draw_logs(cycle_history, inner_height);
            rect.render_widget(logs, tab_layout[1]);
        }
        1 => {
            let plot = draw_plot(&mut norm_data);
            rect.render_widget(plot, tab_layout[1]);
        }
        2 => {
            let par_bounds = draw_parameter_bounds(&settings);
            rect.render_widget(par_bounds, tab_layout[1]);
        }
        _ => unreachable!(),
    };
}
