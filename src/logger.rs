use crate::prelude::output::OutputFile;
use crate::routines::settings::Settings;
use crate::tui::ui::Comm;
use anyhow::Result;
use std::io::{self, Write};
use tokio::sync::mpsc::UnboundedSender;
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
/// Additionally, if the `tui` option is set to `true`, the log messages are also written to the TUI.
///
/// If not, the log messages are written to stdout.
pub fn setup_log(settings: &Settings, ui_tx: Option<UnboundedSender<Comm>>) -> Result<()> {
    // Use the log level defined in configuration file
    let log_level = settings.log.level.as_str();
    let env_filter = EnvFilter::new(log_level);

    // Define a registry with that level as an environment filter
    let subscriber = Registry::default().with(env_filter);

    // Define outputfile
    let outputfile = match settings.log.write {
        true => Some(OutputFile::new(&settings.output.path, &settings.log.file)?),
        false => None,
    };

    // Define layer for file
    let file_layer = match outputfile {
        Some(outputfile) => {
            let file = outputfile.file;

            let layer = fmt::layer()
                .with_writer(file)
                .with_ansi(false)
                .with_timer(CompactTimestamp);

            Some(layer)
        }
        None => None,
    };

    // Define layer for stdout
    let stdout_layer = match settings.config.tui {
        false => {
            let layer = fmt::layer()
                .with_writer(std::io::stdout)
                .with_ansi(true)
                .with_target(false)
                .with_timer(CompactTimestamp);
            Some(layer)
        }
        true => None,
    };

    // Check if ui_tx is Some and clone it for use in the closure
    let tui_layer = if settings.config.tui {
        if let Some(ui_tx) = ui_tx.clone() {
            // Clone the sender outside the closure
            let tui_writer_closure = move || TuiWriter {
                // Use move to capture the cloned sender
                ui_tx: ui_tx.clone(), // Clone the sender for each closure invocation
            };
            let layer = fmt::layer()
                .with_writer(tui_writer_closure)
                .with_ansi(false)
                .with_target(false)
                .with_timer(CompactTimestamp);
            Some(layer)
        } else {
            None
        }
    } else {
        None
    };

    // Combine layers with subscriber
    subscriber
        .with(file_layer)
        .with(stdout_layer)
        .with(tui_layer)
        .init();

    Ok(())
}

#[derive(Clone)]
struct CompactTimestamp;

impl FormatTime for CompactTimestamp {
    fn format_time(
        &self,
        w: &mut tracing_subscriber::fmt::format::Writer<'_>,
    ) -> Result<(), std::fmt::Error> {
        write!(w, "{}", chrono::Local::now().format("%H:%M:%S"))
    }
}

struct TuiWriter {
    ui_tx: UnboundedSender<Comm>,
}

impl Write for TuiWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let msg = String::from_utf8_lossy(buf);
        // Send the message through the channel
        self.ui_tx
            .send(Comm::LogMessage(msg.to_string()))
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "Failed to send log message"))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        // Flushing is not required for this use case
        Ok(())
    }
}
