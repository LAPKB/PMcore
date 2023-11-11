use tracing::Subscriber;
use tracing_subscriber::{fmt, layer::Context, prelude::*, registry::Registry, Layer, EnvFilter};
use tokio::sync::mpsc::UnboundedSender;

use crate::routines::settings::run::Data;
use crate::tui::ui::Comm;

pub fn setup_log(settings: &Data, tx: UnboundedSender<Comm>) {
    let log_level = settings
        .parsed
        .config
        .log_level
        .as_ref()
        .map(|level| level.as_str())
        .unwrap_or("info");

    let env_filter = EnvFilter::new(log_level);
    let format_layer = fmt::layer().compact();

    // Custom RatatuiLogLayer
    let ratatui_log_layer = RatatuiLogLayer::new(tx);

    // Combine layers
    let subscriber = Registry::default()
        .with(env_filter)
        .with(format_layer)
        .with(ratatui_log_layer);

    subscriber.init();
}

// Custom Layer for sending log messages
struct RatatuiLogLayer {
    sender: UnboundedSender<Comm>,
}

impl RatatuiLogLayer {
    pub fn new(sender: UnboundedSender<Comm>) -> Self {
        RatatuiLogLayer {
            sender
        }
    }
}

impl<S: Subscriber> Layer<S> for RatatuiLogLayer {
    fn on_event(&self, event: &tracing::Event<'_>, ctx: Context<S>) {
        let mut buffer = String::new();
        if ctx.event_format(event, &mut buffer).is_ok() {
            // Send the formatted message through Comm
            let _ = self.sender.send(Comm::LogMessage(buffer));
        }
    }
}
