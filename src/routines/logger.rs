use std::time::Instant;

use crate::routines::output::OutputFile;
use crate::routines::settings::Settings;
use anyhow::Result;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::fmt::{self};
use tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt;
use tracing_subscriber::registry::Registry;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Setup logging for the library
///
/// This function sets up logging for the library. It uses the `tracing` crate, and the `tracing-subscriber` crate for formatting.
///
/// The log level is defined in the configuration file, and defaults to `INFO`.
///
/// If `log_out` is specifified in the configuration file, a log file is created with the specified name.
///
/// If not, the log messages are written to stdout.
pub fn setup_log(settings: &Settings) -> Result<()> {
    // Use the log level defined in configuration file
    let log_level = settings.log().level.clone();
    let env_filter = EnvFilter::new(log_level);

    let timestamper = CompactTimestamp {
        start: Instant::now(),
    };

    // Define a registry with that level as an environment filter
    let subscriber = Registry::default().with(env_filter);

    // Define outputfile
    let outputfile = OutputFile::new(&settings.output().path, &settings.log().file)?;

    // Define layer for file
    let file_layer = fmt::layer()
        .with_writer(outputfile.file)
        .with_ansi(false)
        .with_timer(timestamper.clone());

    // Define layer for stdout
    let stdout_layer = fmt::layer()
        .with_writer(std::io::stdout)
        .with_ansi(true)
        .with_target(false)
        .with_timer(timestamper.clone());

    // Try to set the subscriber
    let result = subscriber.with(file_layer).with(stdout_layer).try_init();

    if result.is_err() {
        // If there's already a subscriber, ignore the error
        tracing::warn!("Logger was already initialized");
    }

    Ok(())
}

/// Cleanup logging setup
///
/// This function removes the global logging subscriber that was set up by `setup_log`.
/// It's recommended to call this function when shutting down the application to ensure
/// proper cleanup of logging resources.
pub fn unsubscribe_log() {
    let _ = tracing::subscriber::set_global_default(tracing::subscriber::NoSubscriber::default());
}

#[derive(Clone)]
struct CompactTimestamp {
    start: Instant,
}

impl FormatTime for CompactTimestamp {
    fn format_time(
        &self,
        w: &mut tracing_subscriber::fmt::format::Writer<'_>,
    ) -> Result<(), std::fmt::Error> {
        let elapsed = self.start.elapsed();
        let hours = elapsed.as_secs() / 3600;
        let minutes = (elapsed.as_secs() % 3600) / 60;
        let seconds = elapsed.as_secs() % 60;

        write!(w, "{:02}h {:02}m {:02}s", hours, minutes, seconds)
    }
}
