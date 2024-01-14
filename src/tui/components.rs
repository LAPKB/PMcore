use std::time::Duration;

/// This file contains the different components of the TUI
/// The purpose is to create common components with generic methods
use ratatui::{
    layout::{Alignment, Constraint},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, Block, BorderType, Borders, Cell, Chart, Dataset, GraphType, Paragraph, Row, Table,
        Tabs, Wrap,
    },
};

use super::App;

use crate::prelude::settings::Settings;

pub fn draw_title<'a>() -> Paragraph<'a> {
    Paragraph::new("NPcore Execution")
        .style(Style::default().fg(Color::LightCyan))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::White))
                .border_type(BorderType::Plain),
        )
}

pub fn draw_status<'a>(app: &App, elapsed_time: Duration) -> Table<'a> {
    // Define (formatted) texts
    let cycle_text = format!("{}", app.state.cycle);
    let objf_text = format!("{:.5}", app.state.objf);
    let delta_objf_text = format!("{:.5}", app.state.delta_objf);
    let gamma_text = format!("{:.5}", app.state.gamlam);
    let spp_text = format!("{}", app.state.nspp);
    let time_text = format_time(elapsed_time);
    let conv_text = "Placeholder".to_string();

    // Define the table data
    let data = vec![
        ("Current cycle", cycle_text),
        ("Objective function", objf_text),
        ("Δ Objective function", delta_objf_text),
        ("Gamma/Lambda", gamma_text),
        ("Support points", spp_text),
        ("Elapsed time", time_text),
        ("Convergence", conv_text),
        // Add more rows as needed
    ];

    // Populate the table rows
    let rows: Vec<Row> = data
        .iter()
        .map(|(key, value)| {
            let title_style = Style::default().add_modifier(Modifier::BOLD);
            let title_cell = Cell::from(Span::styled(format!("{}:", key), title_style));
            let value_cell = Cell::from(value.to_string());
            Row::new(vec![title_cell, value_cell])
        })
        .collect();

    // Create the table widget
    Table::default()
        .rows(rows)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Plain)
                .title(" Status "),
        )
        .widths(&[Constraint::Percentage(50), Constraint::Percentage(50)]) // Set percentage widths for columns
        .column_spacing(1)
}

pub fn draw_options<'a>(settings: &Settings) -> Table<'a> {
    // Define the table data

    let cycles = settings.config.cycles.to_string();
    let engine = settings.config.engine.to_string();
    let conv_crit = "Placeholder".to_string();
    let indpts = settings.config.init_points.to_string();
    let error = settings.error.class.to_string();
    let cache = match settings.config.cache {
        true => "Enabled".to_string(),
        false => "Disabled".to_string(),
    };
    let seed = settings.config.seed.to_string();

    let data = vec![
        ("Maximum cycles", cycles),
        ("Engine", engine),
        ("Convergence criteria", conv_crit),
        ("Initial gridpoints", indpts),
        ("Error model", error),
        ("Cache", cache),
        ("Random seed", seed),
        // Add more rows as needed
    ];

    // Populate the table rows
    let rows: Vec<Row> = data
        .iter()
        .map(|(key, value)| {
            let title_style = Style::default().add_modifier(Modifier::BOLD);
            let title_cell = Cell::from(Span::styled(format!("{}:", key), title_style));
            let value_cell = Cell::from(value.to_string());
            Row::new(vec![title_cell, value_cell])
        })
        .collect();

    // Create the table widget
    Table::default()
        .rows(rows)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Plain)
                .title(" Options "),
        )
        .widths(&[Constraint::Percentage(50), Constraint::Percentage(50)]) // Set percentage widths for columns
        .column_spacing(1)
}

pub fn draw_commands(app: &App) -> Table {
    let key_style = Style::default().fg(Color::LightCyan);
    let help_style = Style::default().fg(Color::Gray);

    let mut rows = vec![];
    for action in app.actions.actions().iter() {
        let mut first = true;
        for key in action.keys() {
            let help = if first {
                first = false;
                action.to_string()
            } else {
                String::from("")
            };
            let row = Row::new(vec![
                Cell::from(Span::styled(key.to_string(), key_style)),
                Cell::from(Span::styled(help, help_style)),
            ]);
            rows.push(row);
        }
    }

    Table::default()
        .rows(rows)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Plain)
                .title(" Commands "),
        )
        .widths(&[Constraint::Percentage(50), Constraint::Percentage(50)]) // Set percentage widths for columns
        .column_spacing(1)
}

pub fn draw_plot(norm_data: &mut [(f64, f64)]) -> Chart {
    // Find min and max values
    let (x_min, x_max) = norm_data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (x, _)| {
            (min.min(*x), max.max(*x))
        });

    let (y_min, y_max) = norm_data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (_, y)| {
            (min.min(*y), max.max(*y))
        });

    // Compute the dynamic step size for the X-labels
    let step_size = ((x_max - x_min) / 10.0).max(1.0).ceil();

    // Generate X-labels using the dynamic step size
    let x_labels: Vec<Span> = ((x_min as i64)..=(x_max as i64))
        .step_by(step_size as usize)
        .map(|x| Span::from(x.to_string()))
        .collect();

    // Generate four Y-labels, evenly from y_min to y_max
    let y_step = (y_max - y_min) / 5.0; // To get 4 labels, we need 3 steps
    let y_labels: Vec<Span> = (0..=3)
        .map(|i| {
            let y = y_min + y_step * (i as f64);
            Span::from(format!("{:.0}", y))
        })
        .collect();

    // Prepare the dataset
    let dataset = vec![Dataset::default()
        .name("-2LL")
        .marker(symbols::Marker::Dot)
        .style(Style::default().fg(Color::Cyan))
        .graph_type(GraphType::Scatter)
        .data(norm_data)];

    // Return the plot
    Chart::new(dataset)
        .x_axis(
            Axis::default()
                .title("Cycle")
                .bounds([x_min, x_max])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title("-2LL")
                .bounds([y_min, y_max])
                .labels(y_labels),
        )
        .block(
            Block::default()
                .title(" Objective function ")
                .borders(Borders::ALL),
        )
}

pub fn draw_logs<'a>(log_history: &'a Vec<String>, height: u16) -> Paragraph<'a> {
    // Convert each String in log_history to a Line
    let text: Vec<Line> = log_history.iter().map(|s| Line::from(s.as_str())).collect();

    let to_text = text.len();
    // Prevent underflow with saturating_sub
    let from_text = to_text.saturating_sub(height as usize);

    // Create a slice of the text to be displayed
    let show_text = if from_text < to_text {
        text[from_text..to_text].to_vec()
    } else {
        Vec::new()
    };

    Paragraph::new(show_text)
        .block(Block::default().title(" Logs ").borders(Borders::ALL))
        .style(Style::default().fg(Color::White))
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true })
}

pub fn draw_tabs<'a>(app: &App) -> Tabs<'a> {
    let titles = app.tab_titles.clone();
    let index = app.tab_index.clone();
    let tabs = Tabs::new(titles.clone())
        .block(Block::default().borders(Borders::ALL))
        .style(Style::default().fg(Color::Cyan))
        .highlight_style(Style::default().fg(Color::Yellow))
        .divider(Span::raw("|"))
        .select(index);

    tabs
}

fn format_time(elapsed_time: std::time::Duration) -> String {
    let elapsed_seconds = elapsed_time.as_secs();
    let (elapsed, unit) = if elapsed_seconds < 60 {
        (elapsed_seconds, "s")
    } else if elapsed_seconds < 3600 {
        let elapsed_minutes = elapsed_seconds / 60;
        (elapsed_minutes, "m")
    } else {
        let elapsed_hours = elapsed_seconds / 3600;
        (elapsed_hours, "h")
    };
    let time_text = format!("{}{}", elapsed, unit);
    time_text
}
