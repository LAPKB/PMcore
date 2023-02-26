use std::{cell::RefCell, rc::Rc, io::{stdout}};
use eyre::Result;
use tui::{backend::{CrosstermBackend, Backend}, Terminal, Frame, layout::{Constraint, Direction, Layout, Alignment, Rect}, widgets::{Paragraph, Block, Borders, BorderType}, style::{Style, Color}};

// #[derive(Sync)]
pub struct App{
    pub cycle: usize
}
impl App{
    pub fn new()->Self{App{cycle: 0}}
    
}
pub fn start_ui(app: Rc<RefCell<App>>) -> Result<()>{
    let stdout = stdout();
    crossterm::terminal::enable_raw_mode()?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    terminal.clear()?;

    loop {
        let app = app.borrow();
        terminal.draw(|rect| draw(rect, &app));

    }
    terminal.clear()?;
    terminal.show_cursor()?;
    crossterm::terminal::disable_raw_mode()?;
    Ok(())
}

pub fn draw<B>(rect: &mut Frame<B>, _app: &App)
where
    B: Backend,
{
    let size = rect.size();
    check_size(&size);

    // Vertical layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3)].as_ref())
        .split(size);

    // Title
    let title = draw_title();
    rect.render_widget(title, chunks[0]);
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

fn check_size(rect: &Rect) {
    if rect.width < 52 {
        panic!("Require width >= 52, (got {})", rect.width);
    }
    if rect.height < 12 {
        panic!("Require height >= 12, (got {})", rect.height);
    }
}