use std::fs::{create_dir_all, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use tracing::Level;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, Registry};

/// Default directives appended to the level filter.
///
/// `diffsol` is silenced because it is very verbose at `INFO` and below and
/// rarely useful for end users of PMcore.
const DEFAULT_DIRECTIVES: &str = "diffsol=off";

/// Builder for an opinionated `tracing` subscriber.
///
/// By default the subscriber writes `INFO` level events to stdout and does
/// not write to a file. The subscriber itself does not render timestamps;
/// callers that want elapsed-time information are expected to include it in
/// their log messages, for example using [`format_elapsed`] with a start
/// instant tracked on the algorithm.
#[derive(Debug, Clone)]
pub struct Logger {
    level: Level,
    stdout: bool,
    file: Option<PathBuf>,
    extra_directives: Option<String>,
}

impl Default for Logger {
    fn default() -> Self {
        Self {
            level: Level::INFO,
            stdout: true,
            file: None,
            extra_directives: Some(DEFAULT_DIRECTIVES.to_string()),
        }
    }
}

impl Logger {
    /// Create a new [`Logger`] builder with default settings (stdout at `INFO`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum level of events to record.
    pub fn level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }

    /// Enable or disable the stdout layer. Enabled by default.
    pub fn stdout(mut self, enable: bool) -> Self {
        self.stdout = enable;
        self
    }

    /// Write logs to the given file path. Parent directories are created as
    /// needed and the file is truncated on open.
    pub fn file<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.file = Some(path.as_ref().to_path_buf());
        self
    }

    /// Disable writing logs to a file.
    pub fn no_file(mut self) -> Self {
        self.file = None;
        self
    }

    /// Override the additional `EnvFilter` directives appended after the level.
    ///
    /// Defaults to `"diffsol=off"`. Pass an empty string to disable the
    /// default directives entirely. The `RUST_LOG` environment variable, if
    /// set, takes precedence over the configured level and directives.
    pub fn directives(mut self, directives: impl Into<String>) -> Self {
        let directives = directives.into();
        self.extra_directives = if directives.is_empty() {
            None
        } else {
            Some(directives)
        };
        self
    }

    /// Install the configured subscriber as the global default.
    ///
    /// Returns an error if a log file is configured but cannot be opened.
    /// Silently does nothing if a global subscriber is already installed,
    /// which makes the subscriber safe to reuse across back-to-back runs.
    pub fn init(self) -> Result<()> {
        let env_filter = self.build_env_filter();

        let file_layer = match self.file.as_deref() {
            Some(path) => Some(open_file_layer(path)?),
            None => None,
        };

        let stdout_layer = if self.stdout {
            Some(
                fmt::layer()
                    .with_writer(std::io::stdout)
                    .with_ansi(true)
                    .with_target(false)
                    .without_time()
                    .boxed(),
            )
        } else {
            None
        };

        let _ = Registry::default()
            .with(env_filter)
            .with(file_layer)
            .with(stdout_layer)
            .try_init();

        Ok(())
    }

    fn build_env_filter(&self) -> EnvFilter {
        // Honor RUST_LOG when set, otherwise fall back to the configured level
        // plus any extra directives.
        if std::env::var("RUST_LOG").is_ok() {
            return EnvFilter::from_default_env();
        }

        let mut directives = self.level.to_string().to_lowercase();
        if let Some(extra) = &self.extra_directives {
            directives.push(',');
            directives.push_str(extra);
        }
        EnvFilter::new(directives)
    }
}

fn open_file_layer<S>(
    path: &Path,
) -> Result<Box<dyn tracing_subscriber::Layer<S> + Send + Sync + 'static>>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)
                .with_context(|| format!("failed to create log directory {:?}", parent))?;
        }
    }

    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("failed to open log file {:?}", path))?;

    Ok(fmt::layer()
        .with_writer(file)
        .with_ansi(false)
        .without_time()
        .boxed())
}

/// Format a [`Duration`] as a compact `HHh MMm SSs` string.
///
/// Intended for embedding elapsed-time information into log messages emitted
/// by an algorithm that tracks its own start instant, e.g.:
///
/// ```no_run
/// use std::time::Instant;
/// use pmcore::logs::format_elapsed;
///
/// let start = Instant::now();
/// // ... do work ...
/// tracing::info!("cycle finished in {}", format_elapsed(start.elapsed()));
/// ```
pub fn format_elapsed(elapsed: Duration) -> String {
    let secs = elapsed.as_secs();
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    format!("{:02}h {:02}m {:02}s", hours, minutes, seconds)
}
