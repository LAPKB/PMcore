use tracing_subscriber::fmt::{self, format::Format};
use tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt;
use tracing_subscriber::registry::Registry;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use crate::routines::settings::run::Data;

pub fn setup_log(settings: &Data) {
    let log_level = settings
        .parsed
        .config
        .log_level
        .as_ref()
        .map(|level| level.as_str())
        .unwrap_or("info"); // Default to 'info' if not set

    let env_filter = EnvFilter::new(log_level);

    let stdout_log = Format::default().compact();

    // Start with a base subscriber from the registry
    let subscriber = Registry::default().with(env_filter);

    // Check if a log file path is provided
    if let Some(log_path) = &settings.parsed.paths.log_out {
        // Ensure the log file is created or truncated
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(log_path)
            .expect("Failed to open log file");

        let file_layer = fmt::layer()
            .with_writer(file)
            .with_ansi(false)
            .event_format(stdout_log);

        // Add the file layer to the subscriber
        subscriber.with(file_layer).init();
    } else {
        // Add stdout layer only if no log file is specified
        let stdout_layer = fmt::layer()
            .event_format(stdout_log)
            .with_writer(std::io::stdout);

        // Add the stdout layer to the subscriber
        subscriber.with(stdout_layer).init();
    }

    tracing::info!("Logging is configured with level: {}", log_level);
}
