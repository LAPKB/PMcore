use crate::algorithms::initialize_algorithm;
use crate::prelude::{output::NPResult, *};
use crate::routines::settings::*;

use eyre::Result;
use pharmsol::prelude::data::Data;

use pharmsol::prelude::{data::read_pmetrics, simulator::Equation};
use std::path::Path;
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
pub fn simulate(_equation: Equation, _settings_path: String) -> Result<()> {
    unimplemented!();
    // let settings: Settings = read_settings(settings_path).unwrap();
    // let theta_file = File::open(settings.paths.prior.unwrap()).unwrap();
    // let mut reader = ReaderBuilder::new()
    //     .has_headers(true)
    //     .from_reader(theta_file);
    // // let theta: Array2<f64> = reader.deserialize_array2_dynamic().unwrap();

    // // Expand data
    // // let idelta = settings.config.idelta;
    // // let tad = settings.config.tad;
    // let data = read_pmetrics(Path::new(settings.paths.data.as_str())).unwrap();
    // // let subjects = data.get_subjects();

    // // Perform simulation
    // // let obspred = get_population_predictions(&equation, &subjects, &theta, false);

    // // Prepare writer
    // let sim_file = File::create("simulation_output.csv").unwrap();
    // let mut sim_writer = WriterBuilder::new()
    //     .has_headers(false)
    //     .from_writer(sim_file);
    // sim_writer
    //     .write_record(["id", "point", "time", "pred"])
    //     .unwrap();

    // // Write output
    // // for (id, subject) in subjects.iter().enumerate() {
    // //     //TODO: We are missing a get_obs_times function
    // //     // let time = subject.obs_times.clone();
    // //     let time: Vec<f64> = vec![];
    // //     for (point, _spp) in theta.rows().into_iter().enumerate() {
    // //         for (i, time) in time.iter().enumerate() {
    // //             unimplemented!()
    // //             // sim_writer.write_record(&[
    // //             //     id.to_string(),
    // //             //     point.to_string(),
    // //             //     time.to_string(),
    // //             //     obspred
    // //             //         .get((id, point))
    // //             //         .unwrap()
    // //             //         .get(i)
    // //             //         .unwrap()
    // //             //         .to_string(),
    // //             // ])?;
    // //         }
    // //     }
    // // }
    // Ok(())
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

    tracing::info!("Starting {}", settings.config.engine);

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

    // Initialize algorithm and run
    let mut algorithm = initialize_algorithm(equation.clone(), settings.clone(), data, tx.clone());
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
