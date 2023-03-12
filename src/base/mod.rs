use self::datafile::Scenario;
use self::prob::{sim_obs, Observations};
use self::settings::Data;
use self::simulator::{Engine, Simulate};
use crate::prelude::start_ui;
use crate::{algorithms::npag::npag, tui::state::AppState};
use csv::{ReaderBuilder, WriterBuilder};
use eyre::Result;
use log::LevelFilter;
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;
use ndarray::{Array2, Axis};
use ndarray_csv::{Array2Reader, Array2Writer};
use std::fs::{self, File};
use std::thread::spawn;
use std::time::Instant;
use tokio::sync::mpsc::{self, UnboundedSender};
pub mod array_permutation;
pub mod datafile;
pub mod ipm;
pub mod lds;
pub mod prob;
pub mod settings;
pub mod simulator;

pub fn start<S>(
    engine: Engine<S>,
    ranges: Vec<(f64, f64)>,
    settings_path: String,
    c: (f64, f64, f64, f64),
) -> Result<()>
where
    S: Simulate + std::marker::Sync + std::marker::Send + 'static,
{
    let now = Instant::now();
    let settings = settings::read(settings_path);
    setup_log(&settings);
    let theta = match &settings.paths.prior_dist {
        Some(prior_path) => {
            let file = File::open(prior_path).unwrap();
            let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
            let array_read: Array2<f64> = reader.deserialize_array2_dynamic().unwrap();
            array_read
        }
        None => lds::sobol(settings.config.init_points, &ranges, settings.config.seed),
    };
    let mut scenarios = datafile::parse(&settings.paths.data).unwrap();
    if let Some(exclude) = &settings.config.exclude.clone(){
        for val in exclude{
            dbg!(&val);
            scenarios.remove(val.as_integer().unwrap() as usize);
            
        }
    }
    
    let (tx, rx) = mpsc::unbounded_channel::<AppState>();

    if settings.config.tui {
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
    S: Simulate + std::marker::Sync,
{
    let (theta, _w, _objf, _cycle, _converged) =
        npag(&sim_eng, ranges, theta, scenarios, c, tx, settings);

    if let Some(output) = &settings.config.pmetrics_outputs {
        if *output {
            //theta.csv
            let file = File::create("theta.csv").unwrap();
            let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
            // writer.write_record(&["a", "b"]).unwrap();
            // I need to obtain the parameter names, perhaps from the config file?
            writer.serialize_array2(&theta).unwrap();

            //pred.csv
            let pred = sim_obs(&sim_eng, scenarios, &theta);
            let cycles_file = File::create("pred.csv").unwrap();
            let mut writer = WriterBuilder::new()
                .has_headers(false)
                .from_writer(cycles_file);
            for row in pred.axis_iter(Axis(0)) {
                for elem in row.axis_iter(Axis(0)) {
                    writer.write_field(format!("{}", &elem)).unwrap();
                }
                writer.write_record(None::<&[u8]>).unwrap();
            }
            writer.flush().unwrap();

            //obs.csv
            //time.csv
            let obs_file = File::create("obs.csv").unwrap();
            let time_file = File::create("time.csv").unwrap();
            let mut obs_writer = WriterBuilder::new()
                .has_headers(false)
                .from_writer(obs_file);
            let mut time_writer = WriterBuilder::new()
                .has_headers(false)
                .from_writer(time_file);
            for scenario in scenarios {
                let obs = Observations(scenario.obs_flat.clone());
                let time = Observations(scenario.time_flat.clone());
                obs_writer.write_field(format!("{}", obs)).unwrap();
                obs_writer.write_record(None::<&[u8]>).unwrap();
                time_writer.write_field(format!("{}", time)).unwrap();
                time_writer.write_record(None::<&[u8]>).unwrap();
            }
            obs_writer.flush().unwrap();
            time_writer.flush().unwrap();
        }
    }
}

fn setup_log(settings: &Data) {
    if let Some(log_path) = &settings.paths.log_out {
        if fs::remove_file(log_path).is_ok() {};
        let logfile = FileAppender::builder()
            .encoder(Box::new(PatternEncoder::new("{l} - {m}\n")))
            .build(log_path)
            .unwrap();

        let config = Config::builder()
            .appender(Appender::builder().build("logfile", Box::new(logfile)))
            .build(Root::builder().appender("logfile").build(LevelFilter::Info))
            .unwrap();

        log4rs::init_config(config).unwrap();
    };
}
