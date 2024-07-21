use crate::algorithms::initialize_algorithm;
use crate::prelude::{output::NPResult, *};
use crate::routines::settings::*;

use csv::WriterBuilder;
use eyre::Result;
use output::create_output_file;
use pharmsol::prelude::data::Data;

use pharmsol::prelude::{data::read_pmetrics, simulator::Equation};
use std::path::Path;
use std::thread::spawn;
use std::time::Instant;
use tokio::sync::mpsc::{self};

/// Simulate predictions from a model and prior distribution
pub fn simulate(equation: Equation, settings: Settings) -> Result<()> {
    let now = Instant::now();

    // Setup log
    logger::setup_log(&settings, None);

    // Read input data
    let data = read_pmetrics(Path::new(settings.paths.data.as_str())).unwrap();
    let subjects = data.get_subjects();

    // Expand data
    data.expand(settings.config.idelta, settings.config.tad);

    tracing::info!("Preparing simulator...");

    // Provide information of the input data
    tracing::info!(
        // "Datafile contains {} subjects with a total of {} observations",
        "Datafile contains {} subjects with a total of {} occasions",
        subjects.len(),
        subjects.iter().map(|s| s.occasions().len()).sum::<usize>()
    );

    // Read prior
    let ranges = settings.random.ranges();
    let prior = sample_space(&settings, &ranges, &data, &equation);

    // Tell the user where the output file will be written
    tracing::info!(
        "Simulated predictions will be written to {}",
        settings.paths.output_folder.as_ref().unwrap().clone()
    );
    let file = create_output_file(&settings, "sim.csv")?;
    let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);
    writer.write_record(&["subject", "time", "outeq", "pred"])?;

    // Perform simulation
    for subject in data.get_subjects() {
        tracing::info!("Simulating subject {}", subject.id());
        for support_point in prior.rows() {
            let subject_prediction = equation.simulate_subject(subject, &support_point.to_vec());

            for pred in subject_prediction.get_predictions() {
                writer.write_record(&[
                    &subject.id(),
                    &pred.time().to_string(),
                    &pred.outeq().to_string(),
                    &pred.prediction().to_string(),
                ])?;
            }
        }
    }
    writer.flush()?;

    tracing::info!("Simulation complete after {:.2?}", now.elapsed());
    Ok(())
}

/// Primary entrypoint for PMcore
///
/// This function is the primary entrypoint for PMcore, and is used to run the algorithm.
/// The settings for this function is specified in a TOML configuration file, see `routines::settings::run` for details.
pub fn fit(equation: Equation, settings: Settings) -> Result<NPResult> {
    let now = Instant::now();

    // Configure MPSC channels for TUI
    let (tx, rx) = match settings.config.tui {
        true => {
            let (to, from) = mpsc::unbounded_channel::<Comm>();
            (Some(to), Some(from))
        }
        false => (None, None),
    };

    logger::setup_log(&settings, tx.clone());
    tracing::info!("Starting PMcore");

    // Read input data
    let data = read_pmetrics(Path::new(settings.paths.data.as_str())).unwrap();
    let subjects = data.get_subjects();
    // Provide information of the input data
    tracing::info!(
        // "Datafile contains {} subjects with a total of {} observations",
        "Datafile contains {} subjects with a total of {} occasions",
        subjects.len(),
        //TODO: again we are missing a get_obs_times function
        // subjects.iter().map(|s| s.obs_times.len()).sum::<usize>()
        subjects.iter().map(|s| s.occasions().len()).sum::<usize>()
    );

    // Tell the user where the output files will be written
    tracing::info!(
        "Output files will be written to {}",
        settings.paths.output_folder.as_ref().unwrap().clone()
    );

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
    let mut algorithm = initialize_algorithm(equation.clone(), settings.clone(), data, tx.clone());

    // Tell the user which algorithm is being used
    tracing::info!(
        "The program will run with the {} algorithm",
        settings.config.engine
    );

    // Run the algorithm
    let result = algorithm.fit();

    // Write output files (if configured)
    if settings.config.output {
        let idelta = settings.config.idelta;
        let tad = settings.config.tad;
        result.write_outputs(true, &equation, idelta, tad);
    }

    if let Some(tx) = tx {
        tx.send(Comm::StopUI).unwrap();
    }
    handle.join().unwrap();
    tracing::info!("Program complete after {:.2?}", now.elapsed());

    Ok(result)
}

/// Alternative entrypoint, primarily meant for third-party libraries or APIs
///
/// This entrypoint takes an `Engine` (from the model), `Data` from the settings, and `scenarios` containing dose information and observations
///
/// It does not write any output files, and does not start a TUI.
///
/// Returns an NPresult object
pub fn start_internal(equation: Equation, settings: Settings, data: Data) -> Result<NPResult> {
    let now = Instant::now();
    logger::setup_log(&settings, None);

    let mut algorithm = initialize_algorithm(equation, settings.clone(), data, None);

    let result = algorithm.fit();
    tracing::info!("Total time: {:.2?}", now.elapsed());
    Ok(result)
}
