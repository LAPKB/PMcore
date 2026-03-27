use std::time::Instant;

use anyhow::Result;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::fmt::{self};
use tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt;
use tracing_subscriber::registry::Registry;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use crate::api::{LoggingLevel, LoggingOptions, OutputPlan};
use crate::output::OutputFile;

pub(crate) fn setup_log_with_options(output: &OutputPlan, logging: &LoggingOptions) -> Result<()> {
    let log_level = log_level_filter(logging.level);
    let env_filter = EnvFilter::new(format!("{},diffsol=off", log_level));

    if !logging.stdout && !logging.write {
        let subscriber = Registry::default().with(env_filter);
        let _ = subscriber.try_init();
        return Ok(());
    }

    let timestamper = CompactTimestamp {
        start: Instant::now(),
    };

    let subscriber = Registry::default().with(env_filter);

    let outputfile = if !output.write {
        let cd = std::env::current_dir()?;
        OutputFile::new(&cd.to_string_lossy(), "log.txt")?
    } else {
        OutputFile::new(output.path.as_deref().unwrap_or("outputs/"), "log.txt")?
    };

    let file_layer = match logging.write {
        true => {
            let layer = fmt::layer()
                .with_writer(outputfile.file_owned())
                .with_ansi(false)
                .with_timer(timestamper.clone());

            Some(layer)
        }
        false => None,
    };

    let stdout_layer = match logging.stdout {
        true => {
            let layer = fmt::layer()
                .with_writer(std::io::stdout)
                .with_ansi(true)
                .with_target(false)
                .with_timer(timestamper.clone());

            Some(layer)
        }
        false => None,
    };

    let _ = subscriber.with(file_layer).with(stdout_layer).try_init();

    Ok(())
}

fn log_level_filter(level: LoggingLevel) -> &'static str {
    match level {
        LoggingLevel::Trace => "trace",
        LoggingLevel::Debug => "debug",
        LoggingLevel::Info => "info",
        LoggingLevel::Warn => "warn",
        LoggingLevel::Error => "error",
    }
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