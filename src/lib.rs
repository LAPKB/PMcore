pub mod algorithms;
pub mod routines {
    pub mod datafile;
    pub mod initialization {
        pub mod sobol;
    }
    pub mod optimization {
        pub mod expansion;
        pub mod optim;
    }
    pub mod output;

    pub mod settings {
        pub mod run;
        pub mod simulator;
    }
    pub mod evaluation {
        pub mod ipm;
        pub mod prob;
        pub mod qr;
        pub mod sigma;
    }
    pub mod simulation {
        pub mod predict;
        pub mod simulator;
    }
}
pub mod tui;

pub mod prelude {
    pub use crate::algorithms;
    pub use crate::prelude::evaluation::{prob, sigma, *};
    pub use crate::routines::initialization::*;
    pub use crate::routines::optimization::*;
    pub use crate::routines::simulation::*;
    pub use crate::routines::*;
    pub use crate::tui::ui::*;
}

use eyre::Result;
use log::LevelFilter;
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;
use ndarray::Array2;

use prelude::algorithms::initialize_algorithm;
use prelude::{
    datafile::Scenario,
    output::{NPCycle, NPResult},
    predict::{Engine, Predict},
    settings::run::Data,
    *,
};
use std::fs;
use std::thread::spawn;
use std::{fs::File, time::Instant};
use tokio::sync::mpsc::{self, UnboundedSender};
//Tests
mod tests;

pub fn start<S>(engine: Engine<S>, settings_path: String) -> Result<NPResult>
where
    S: Predict + std::marker::Sync + std::marker::Send + 'static + Clone,
{
    let now = Instant::now();
    let settings = settings::run::read(settings_path);
    setup_log(&settings);
    let ranges = settings.computed.random.ranges.clone();
    let theta = match &settings.parsed.paths.prior_dist {
        Some(prior_path) => {
            let file = File::open(prior_path).unwrap();
            let mut reader = csv::ReaderBuilder::new()
                .has_headers(true)
                .from_reader(file);

            let mut parameter_names: Vec<String> = reader
                .headers()
                .unwrap()
                .clone()
                .into_iter()
                .map(|s| s.trim().to_owned())
                .collect();

            // Remove "prob" column if present
            if let Some(index) = parameter_names.iter().position(|name| name == "prob") {
                parameter_names.remove(index);
            }

            // Check and reorder parameters to match names in settings.parsed.random
            let random_names: Vec<String> = settings
                .parsed
                .random
                .iter()
                .map(|(name, _)| name.clone())
                .collect();

            let mut reordered_indices: Vec<usize> = Vec::new();
            for random_name in &random_names {
                match parameter_names.iter().position(|name| name == random_name) {
                    Some(index) => {
                        reordered_indices.push(index);
                    }
                    None => {
                        panic!("Parameter {} is not present in the CSV file.", random_name);
                    }
                }
            }

            // Check if there are remaining parameters not present in settings.parsed.random
            if parameter_names.len() > random_names.len() {
                let extra_parameters: Vec<&String> = parameter_names.iter().collect();
                panic!(
                    "Found parameters in the prior not present in configuration: {:?}",
                    extra_parameters
                );
            }

            // Read parameter values row by row, keeping only those associated with the reordered parameters
            let mut theta_values = Vec::new();
            for result in reader.records() {
                let record = result.unwrap();
                let values: Vec<f64> = reordered_indices
                    .iter()
                    .map(|&i| record[i].parse::<f64>().unwrap())
                    .collect();
                theta_values.push(values);
            }

            let n_points = theta_values.len();
            let n_params = random_names.len();

            // Convert nested Vec into a single Vec
            let theta_values: Vec<f64> = theta_values.into_iter().flatten().collect();

            Array2::from_shape_vec((n_points, n_params), theta_values)
                .expect("Failed to create theta Array2")
        }
        None => sobol::generate(
            settings.parsed.config.init_points,
            &ranges,
            settings.parsed.config.seed,
        ),
    };

    let mut scenarios = datafile::parse(&settings.parsed.paths.data).unwrap();
    if let Some(exclude) = &settings.parsed.config.exclude {
        for val in exclude {
            scenarios.remove(val.as_integer().unwrap() as usize);
        }
    }

    let (tx, rx) = mpsc::unbounded_channel::<NPCycle>();
    let c = settings.parsed.error.poly;

    let settings_tui = settings.clone();

    let npag_result: NPResult;

    if settings.parsed.config.tui {
        let ui_handle = spawn(move || {
            start_ui(rx, settings_tui).expect("Failed to start UI");
        });

        npag_result = run_npag(engine, ranges, theta, &scenarios, c, tx, &settings);
        log::info!("Total time: {:.2?}", now.elapsed());
        ui_handle.join().expect("UI thread panicked");
    } else {
        npag_result = run_npag(engine, ranges, theta, &scenarios, c, tx, &settings);
        log::info!("Total time: {:.2?}", now.elapsed());
    }

    Ok(npag_result)
}

fn run_npag<S>(
    sim_eng: Engine<S>,
    ranges: Vec<(f64, f64)>,
    theta: Array2<f64>,
    scenarios: &Vec<Scenario>,
    c: (f64, f64, f64, f64),
    tx: UnboundedSender<NPCycle>,
    settings: &Data,
) -> NPResult
where
    S: Predict + std::marker::Sync + std::marker::Send + 'static + Clone,
{
    // Remove stop file if exists
    if std::path::Path::new("stop").exists() {
        match std::fs::remove_file("stop") {
            Ok(_) => log::info!("Removed previous stop file"),
            Err(err) => panic!("Unable to remove previous stop file: {}", err),
        }
    }

    // let algorithm = NPAG::<S>::initialize(
    //     sim_eng,
    //     ranges,
    //     theta,
    //     scenarios.to_owned(),
    //     c,
    //     tx,
    //     settings.clone(),
    // );
    let mut algorithm = initialize_algorithm(
        algorithms::Type::NPAG,
        sim_eng,
        ranges,
        theta,
        scenarios.clone(),
        c,
        tx,
        settings.clone(),
    );
    let (sim_eng, result) = algorithm.fit();

    // let result = npag(&sim_eng, ranges, theta, scenarios, c, tx, settings);

    if let Some(output) = &settings.parsed.config.pmetrics_outputs {
        if *output {
            result.write_theta();
            result.write_posterior();
            result.write_obs();
            result.write_pred(&sim_eng);
            result.write_meta();
        }
    }

    result
}

//TODO: move elsewhere
fn setup_log(settings: &Data) {
    if let Some(log_path) = &settings.parsed.paths.log_out {
        if fs::remove_file(log_path).is_ok() {};
        let logfile = FileAppender::builder()
            .encoder(Box::new(PatternEncoder::new("{l}: {m}\n")))
            .build(log_path)
            .unwrap();

        let config = Config::builder()
            .appender(Appender::builder().build("logfile", Box::new(logfile)))
            .build(Root::builder().appender("logfile").build(LevelFilter::Info))
            .unwrap();

        log4rs::init_config(config).unwrap();
    };
}
