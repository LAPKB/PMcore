//! Out-of-band stopping of a running fit.
//!
//! PMcore follows the classic NONMEM/Pmetrics convention of a `stop` file in the
//! working directory: creating one asks the running fit to stop after the
//! current cycle. [`FitController::request_stop`](crate::algorithms::nonparametric::FitController::request_stop)
//! is the in-process equivalent.

use std::path::Path;

use anyhow::{Context, Result};

/// Name of the file whose presence asks a running fit to stop.
const STOP_FILE: &str = "stop";

/// Whether a stop has been requested through the stop file.
pub(crate) fn stop_file_present() -> bool {
    Path::new(STOP_FILE).exists()
}

/// Remove a stale stop file left over from a previous run.
pub(crate) fn remove_stop_file() -> Result<()> {
    if stop_file_present() {
        tracing::info!("Removing existing stop file prior to run");
        std::fs::remove_file(STOP_FILE).context("Unable to remove previous stop file")?;
    }

    Ok(())
}
