use self::datafile::Scenario;
use self::output_statistics::{population_mean_median, posterior, posterior_mean_median};
use self::predict::{post_predictions, Engine, Predict};
use self::settings::Data;
use crate::prelude::start_ui;
use crate::{algorithms::npag::npag, tui::state::AppState};
use csv::{ReaderBuilder, WriterBuilder};
use eyre::Result;
use log::LevelFilter;
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;

use ndarray::{Array2, Axis};
use ndarray_csv::Array2Reader;
// use ndarray_csv::Array2Writer;
use predict::sim_obs;

use std::fs::{self, File};
use std::thread::spawn;
use std::time::Instant;
use tokio::sync::mpsc::{self, UnboundedSender};
pub mod array_permutation;
pub mod datafile;
pub mod ipm;
pub mod lds;
pub mod optim;
pub mod output_statistics;
pub mod predict;
pub mod prob;
pub mod settings;
pub mod sigma;

pub fn start<S>(engine: Engine<S>, settings_path: String) -> Result<()>
where
    S: Predict + std::marker::Sync + std::marker::Send + 'static,
{
    let now = Instant::now();
    let settings = settings::read(settings_path);
    setup_log(&settings);
    let ranges = settings.computed.random.ranges.clone();
    let theta = match &settings.parsed.paths.prior_dist {
        Some(prior_path) => {
            let file = File::open(prior_path).unwrap();
            let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
            let array_read: Array2<f64> = reader.deserialize_array2_dynamic().unwrap();
            array_read
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
    let (tx, rx) = mpsc::unbounded_channel::<AppState>();
    let c = settings.parsed.error.poly;

    if settings.parsed.config.tui {
        spawn(move || {
            run_npag(engine, ranges, theta, &scenarios, c, tx, &settings);
            log::info!("Total time: {:.2?}", now.elapsed());
        });
        start_ui(rx)?;
    } else {
        run_npag(engine, ranges, theta, &scenarios, c, tx, &settings);
        log::info!("Total time: {:.2?}", now.elapsed());
    }
    Ok(())
}

fn run_npag<S>(
    sim_eng: Engine<S>,
    ranges: Vec<(f64, f64)>,
    theta: Array2<f64>,
    scenarios: &Vec<Scenario>,
    c: (f64, f64, f64, f64),
    tx: UnboundedSender<AppState>,
    settings: &Data,
) where
    S: Predict + std::marker::Sync,
{
    // Remove stop file if exists
    let filename = "stop";
    let _ = std::fs::remove_file(filename);

    let (theta, psi, w, _objf, _cycle, _converged) =
        npag(&sim_eng, ranges, theta, scenarios, c, tx, settings);

    if let Some(output) = &settings.parsed.config.pmetrics_outputs {
        if *output {
            //theta.csv
            let file = File::create("theta.csv").unwrap();
            let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
            // writer.write_record(&["a", "b"]).unwrap();
            // I need to obtain the parameter names, perhaps from the config file?
            let mut theta_w = theta.clone();
            theta_w.push_column(w.view()).unwrap();
            for row in theta_w.axis_iter(Axis(0)) {
                for elem in row.axis_iter(Axis(0)) {
                    writer.write_field(format!("{}", &elem)).unwrap();
                }
                writer.write_record(None::<&[u8]>).unwrap();
            }
            writer.flush().unwrap();

            // // posterior.csv
            let posterior = posterior(&psi, &w);
            let post_file = File::create("posterior.csv").unwrap();
            let mut post_writer = WriterBuilder::new()
                .has_headers(false)
                .from_writer(post_file);
            post_writer.write_field("id").unwrap();
            post_writer.write_field("point").unwrap();
            let parameter_names = &settings.computed.random.names;
            for i in 0..theta.ncols() {
                let param_name = parameter_names.get(i).unwrap();
                post_writer.write_field(param_name).unwrap();
            }
            post_writer.write_field("prob").unwrap();
            post_writer.write_record(None::<&[u8]>).unwrap();

            for (sub, row) in posterior.axis_iter(Axis(0)).enumerate() {
                for (spp, elem) in row.axis_iter(Axis(0)).enumerate() {
                    post_writer
                        .write_field(format!("{}", scenarios.get(sub).unwrap().id))
                        .unwrap();
                    post_writer.write_field(format!("{}", spp)).unwrap();
                    for param in theta.row(spp) {
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
            let (pop_mean, pop_median) = population_mean_median(&theta, &w);
            let (post_mean, post_median) = posterior_mean_median(&theta, &psi, &w);
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
