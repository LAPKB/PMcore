use crate::algorithms::initialize_algorithm;
use crate::prelude::{output::NPResult, *};
use crate::routines::settings::*;

use anyhow::{bail, Context, Result};

use std::thread::spawn;
use std::time::Instant;
use tokio::sync::mpsc::{self};

// use self::simulator::likelihood::Prediction;

/// Simulate predictions from a model and prior distribution
///
/// This function is used to simulate predictions from a model and prior distribution.
/// The output is a CSV file with the following columns:
/// - `id`: subject ID, corresponding to the desired dose regimen
/// - `point`: support point index (0-indexed)
/// - `time`: prediction time
/// - `pred`: simulated prediction
///
/// # Arguments
/// The user can specify the desired settings in a TOML configuration file, see `routines::settings::simulator` for details.
/// - `idelta`: the interval between predictions. Default is 0.0.
/// - `tad`: the time after dose, which if greater than the last prediction time is the time for which it will predict . Default is 0.0.
pub fn simulate(_equation: impl Equation, _data: Data, _settings: Settings) -> Result<()> {
    unimplemented!();
}

/// Primary entrypoint for PMcore
///
/// This function is the primary entrypoint for PMcore, and is used to run the algorithm.
/// The settings for this function is specified in a TOML configuration file, see `routines::settings::run` for details.
pub fn fit(equation: impl Equation, data: Data, settings: Settings) -> anyhow::Result<NPResult> {
    let now = Instant::now();

    // Configure MPSC channels for TUI
    let (tx, rx) = match settings.config.tui {
        true => {
            let (to, from) = mpsc::unbounded_channel::<Comm>();
            (Some(to), Some(from))
        }
        false => (None, None),
    };

    logger::setup_log(&settings, tx.clone())?;
    tracing::info!("Starting PMcore");

    // Read input data
    let subjects = data.get_subjects();

    // Provide information of the input data
    tracing::info!(
        "Datafile contains {} subjects with a total of {} occasions",
        subjects.len(),
        subjects.iter().map(|s| s.occasions().len()).sum::<usize>()
    );

    // Tell the user where the output files will be written
    match &settings.output.write {
        true => {
            let output_path = &settings.output.path;
            tracing::info!("Output files will be written to {}", output_path)
        }
        false => {
            tracing::info!("Output files will not be written - set `output = true` in the configuration file to enable output files")
        }
    }

    // Spawn new thread for TUI
    let settings_tui = settings.clone();
    let handle = if settings.config.tui {
        spawn(move || {
            start_ui(rx.unwrap(), settings_tui).expect("Failed to start TUI");
        })
    } else {
        // Drop messages if TUI is not enabled to reduce memory usage
        spawn(move || {})
    };

    // Initialize algorithm
    let mut algorithm =
        match initialize_algorithm(equation.clone(), settings.clone(), data, tx.clone()) {
            Ok(algorithm) => algorithm,
            Err(err) => {
                tracing::error!(
                    "An error has occurred during algorithm initialization: {}",
                    err
                );
                bail!(
                    "An error has occurred during algorithm initialization: {}",
                    err
                );
            }
        };

    // Tell the user which algorithm is being used
    tracing::info!(
        "The program will run with the {} algorithm",
        settings.config.algorithm
    );

    // Run the algorithm
    let result = match algorithm.fit() {
        Ok(result) => {
            match result.write_outputs(&equation) {
                Ok(_) => {
                    tracing::info!("Output files written successfully");
                }
                Err(err) => {
                    tracing::error!("An error has occurred while writing output files: {}", err);
                }
            };
            Ok(result)
        }
        Err((error, partial_result)) => {
            tracing::error!("An error has occurred during model fitting: {}", error);
            tracing::error!(
                "Attempting to write output files. These can be used to restart the model."
            );
            if settings.output.write {
                if let Err(err) = partial_result.write_outputs(&equation) {
                    tracing::error!("An error has occurred while writing output files: {}", err);
                } else {
                    tracing::info!("Output files written successfully");
                }
            }
            Err(error)
        }
    };

    if let Some(tx) = tx {
        tx.send(Comm::StopUI)
            .context("Failed to send stop signal to the TUI")?;
    }
    handle.join().expect("Failed to close the TUI thread");

    // Provide information about the program runtime|
    tracing::info!("Program complete after {:.2?}", now.elapsed());

    Ok(result?)
}
