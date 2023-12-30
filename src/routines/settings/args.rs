use clap::Parser;

/// My Application
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Number of cycles
    #[arg(short, long)]
    pub cycles: Option<usize>,

    /// Enable text-based user interface
    #[arg(short, long)]
    pub tui: Option<bool>,
}

pub fn parse_args() -> Args {
    Args::parse()
}
