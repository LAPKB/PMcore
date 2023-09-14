use self::datafile::Scenario;
use self::output::{population_mean_median, posterior, posterior_mean_median, NPCycle, NPResult};
use self::predict::{post_predictions, Engine, Predict};
use self::settings::run::Data;
use crate::algorithms::npag::npag;
use crate::prelude::start_ui;
use csv::WriterBuilder;
use eyre::Result;
use log::LevelFilter;
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;
use ndarray::{Array2, Axis};

use predict::sim_obs;
use std::fs::{self, File};
use std::thread::spawn;
use std::time::Instant;
use tokio::sync::mpsc::{self, UnboundedSender};
pub mod array_permutation;
pub mod datafile;
pub mod ipm;
pub mod lds;
pub mod linalg;
pub mod optim;
pub mod output;
pub mod predict;
pub mod prob;
pub mod settings;
pub mod sigma;
pub mod simulator;

pub fn start<S>(engine: Engine<S>, settings_path: String) -> Result<NPResult>
where
    S: Predict + std::marker::Sync + std::marker::Send + 'static,
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
        None => lds::sobol(
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
    S: Predict + std::marker::Sync,
{
    // Remove stop file if exists
    if std::path::Path::new("stop").exists() {
        match std::fs::remove_file("stop") {
            Ok(_) => log::info!("Removed previous stop file"),
            Err(err) => panic!("Unable to remove previous stop file: {}", err),
        }
    }

    let result = npag(&sim_eng, ranges, theta, scenarios, c, tx, settings);

    if let Some(output) = &settings.parsed.config.pmetrics_outputs {
        if *output {
            //theta.csv
            let file = File::create("theta.csv").unwrap();
            let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

            let random_names: Vec<&str> = settings
                .parsed
                .random
                .iter()
                .map(|(name, _)| name.as_str())
                .collect();

            let mut theta_header = random_names.to_vec();
            theta_header.push("prob");

            writer.write_record(&theta_header).unwrap();

            let mut theta_w = result.theta.clone();
            theta_w.push_column(result.w.view()).unwrap();

            for row in theta_w.axis_iter(Axis(0)) {
                for elem in row.axis_iter(Axis(0)) {
                    writer.write_field(format!("{}", &elem)).unwrap();
                }
                writer.write_record(None::<&[u8]>).unwrap();
            }

            writer.flush().unwrap();

            // // posterior.csv
            let posterior = posterior(&result.psi, &result.w);
            let post_file = File::create("posterior.csv").unwrap();
            let mut post_writer = WriterBuilder::new()
                .has_headers(false)
                .from_writer(post_file);
            post_writer.write_field("id").unwrap();
            post_writer.write_field("point").unwrap();
            let parameter_names = &settings.computed.random.names;
            for i in 0..result.theta.ncols() {
                let param_name = parameter_names.get(i).unwrap();
                post_writer.write_field(param_name).unwrap();
            }
            post_writer.write_field("prob").unwrap();
            post_writer.write_record(None::<&[u8]>).unwrap();

            for (sub, row) in posterior.axis_iter(Axis(0)).enumerate() {
                for (spp, elem) in row.axis_iter(Axis(0)).enumerate() {
                    post_writer
                        .write_field(&scenarios.get(sub).unwrap().id)
                        .unwrap();
                    post_writer.write_field(format!("{}", spp)).unwrap();
                    for param in result.theta.row(spp) {
                        post_writer.write_field(format!("{param}")).unwrap();
                    }
                    post_writer.write_field(format!("{elem:.10}")).unwrap();
                    post_writer.write_record(None::<&[u8]>).unwrap();
                }
            }
            // let file = File::create("posterior.csv").unwrap();
            // let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
            // writer.serialize_array2(&posterior).unwrap();
            let cache = settings.parsed.config.cache.unwrap_or(false);
            // pred.csv
            let (pop_mean, pop_median) = population_mean_median(&result.theta, &result.w);
            let (post_mean, post_median) =
                posterior_mean_median(&result.theta, &result.psi, &result.w);
            let post_mean_pred = post_predictions(&sim_eng, post_mean, scenarios).unwrap();
            let post_median_pred = post_predictions(&sim_eng, post_median, scenarios).unwrap();

            let ndim = pop_mean.len();
            let pop_mean_pred = sim_obs(
                &sim_eng,
                scenarios,
                &pop_mean.into_shape((1, ndim)).unwrap(),
                cache,
            );
            let pop_median_pred = sim_obs(
                &sim_eng,
                scenarios,
                &pop_median.into_shape((1, ndim)).unwrap(),
                cache,
            );

            // dbg!(&pop_mean_pred);
            let pred_file = File::create("pred.csv").unwrap();
            let mut pred_writer = WriterBuilder::new()
                .has_headers(false)
                .from_writer(pred_file);
            pred_writer
                .write_record([
                    "id",
                    "time",
                    "outeq",
                    "popMean",
                    "popMedian",
                    "postMean",
                    "postMedian",
                ])
                .unwrap();
            for (id, scenario) in scenarios.iter().enumerate() {
                let time = scenario.obs_times.clone();
                let pop_mp = pop_mean_pred.get((id, 0)).unwrap().to_owned();
                let pop_medp = pop_median_pred.get((id, 0)).unwrap().to_owned();
                let post_mp = post_mean_pred.get(id).unwrap().to_owned();
                let post_mdp = post_median_pred.get(id).unwrap().to_owned();
                for ((((pop_mp_i, pop_mdp_i), post_mp_i), post_medp_i), t) in pop_mp
                    .into_iter()
                    .zip(pop_medp)
                    .zip(post_mp)
                    .zip(post_mdp)
                    .zip(time)
                {
                    pred_writer
                        .write_record(&[
                            scenarios.get(id).unwrap().id.to_string(),
                            t.to_string(),
                            "1".to_string(),
                            pop_mp_i.to_string(),
                            pop_mdp_i.to_string(),
                            post_mp_i.to_string(),
                            post_medp_i.to_string(),
                        ])
                        .unwrap();
                }
            }

            //obs.csv
            let obs_file = File::create("obs.csv").unwrap();
            let mut obs_writer = WriterBuilder::new()
                .has_headers(false)
                .from_writer(obs_file);
            obs_writer
                .write_record(["id", "time", "obs", "outeq"])
                .unwrap();
            for (id, scenario) in scenarios.iter().enumerate() {
                let observations = scenario.obs.clone();
                let time = scenario.obs_times.clone();

                for (obs, t) in observations.into_iter().zip(time) {
                    obs_writer
                        .write_record(&[
                            scenarios.get(id).unwrap().id.to_string(),
                            t.to_string(),
                            obs.to_string(),
                            "1".to_string(),
                        ])
                        .unwrap();
                }
            }
            obs_writer.flush().unwrap();
        }
    }

    result
}

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
