use std::{io::stdout, time::Duration};
use eyre::Result;
use tokio::sync::mpsc::UnboundedReceiver;
use tui::{backend::{CrosstermBackend, Backend}, Terminal, Frame, layout::{Constraint, Direction, Layout, Alignment, Rect}, widgets::{Paragraph, Block, Borders, BorderType, Table, Row, Cell}, style::{Style, Color}, text::{Spans, Span}};

use super::{state::AppState, inputs::{events::Events, InputEvent}, AppReturn, App};


pub fn start_ui(mut rx: UnboundedReceiver<AppState>) -> Result<()>{
    let stdout = stdout();
    crossterm::terminal::enable_raw_mode()?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::new();

    terminal.clear()?;

    // User event handler
    let tick_rate = Duration::from_millis(200);
    let mut events = Events::new(tick_rate);

    loop {
        app.state = match rx.try_recv(){
            Ok(state) => {
                state
            },
            Err(_) => app.state
        };

        terminal.draw(|rect| draw(rect, &app)).unwrap();

        //  Handle inputs
        let result = match events.recv() {
            Some(InputEvent::Input(key)) => app.do_action(key),
            None => AppReturn::Continue
        };
        // Check if we should exit
        if result == AppReturn::Exit {
            break;
        }

    }
    terminal.clear()?;
    terminal.show_cursor()?;
    crossterm::terminal::disable_raw_mode()?;
    Ok(())
}

pub fn draw<B>(rect: &mut Frame<B>, app: &App)
where
    B: Backend,
{
    let size = rect.size();
    check_size(&size);

    // Vertical layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(10)].as_ref())
        .split(size);

    // Title
    let title = draw_title();
    rect.render_widget(title, chunks[0]);

    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(20), Constraint::Length(32)].as_ref())
        .split(chunks[1]);

    let body = draw_body(false, app);
    rect.render_widget(body, body_chunks[0]);

    let help = draw_help(app);
    rect.render_widget(help, body_chunks[1]);
}

fn draw_title<'a>() -> Paragraph<'a> {
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
fn draw_body<'a>(loading: bool, app: &App) -> Paragraph<'a> {
    let loading_text = if loading { "Loading..." } else { "" };
    let cycle_text = format!("Cycle: {}", app.state.cycle);
    let objf_text = format!("-2LL: {}", app.state.objf);
    Paragraph::new(vec![
        Spans::from(Span::raw(loading_text)),
        Spans::from(Span::raw(cycle_text)),
        Spans::from(Span::raw(objf_text)),
    ])
    .style(Style::default().fg(Color::LightCyan))
    .alignment(Alignment::Left)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::White))
            .border_type(BorderType::Plain),
    )
}

fn draw_help(app: &App) -> Table {
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

    Table::new(rows)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Plain)
                .title("Help"),
        )
        .widths(&[Constraint::Length(11), Constraint::Min(20)])
        .column_spacing(1)
}

fn check_size(rect: &Rect) {
    if rect.width < 52 {
        panic!("Require width >= 52, (got {})", rect.width);
    }
    if rect.height < 12 {
        panic!("Require height >= 12, (got {})", rect.height);
    }
}