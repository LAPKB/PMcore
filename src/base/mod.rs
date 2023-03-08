use eyre::Result;
use ndarray::Array2;
use tokio::sync::mpsc::{self, UnboundedSender};
use std::fs::{File, self};
use std::thread;
use ndarray_csv::{Array2Reader, Array2Writer};
// use ndarray_csv::Array2Writer;
use csv::{ReaderBuilder, WriterBuilder};
use crate::prelude::start_ui;
// use csv::WriterBuilder;
use crate::{tui::state::AppState, algorithms::npag::npag};
use log::LevelFilter;
use log4rs::append::file::FileAppender;
use log4rs::encode::pattern::PatternEncoder;
use log4rs::config::{Appender, Config, Root};
use self::datafile::Scenario;
use self::settings::Data;
use self::simulator::{Engine, Simulate};

pub mod settings;
pub mod lds;
pub mod datafile;
pub mod simulator;
pub mod prob;
pub mod ipm;
pub mod array_extra;


pub fn start<S>(engine: Engine<S>, ranges: Vec<(f64, f64)>, settings_path: String, c: (f64,f64,f64,f64)) -> Result<()>
where
S: Simulate + std::marker::Sync + std::marker::Send + 'static
{
    let settings = settings::read(settings_path);
    setup_log(&settings);
    let theta = match &settings.paths.prior_dist {
        Some(prior_path) => {
            let file = File::open(prior_path).unwrap();
            let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
            let array_read: Array2<f64> = reader.deserialize_array2_dynamic().unwrap();
            array_read
        },
        None => lds::sobol(settings.config.init_points, &ranges, settings.config.seed)

    };
    let scenarios = datafile::parse(&settings.paths.data).unwrap();
    let (tx, rx) = mpsc::unbounded_channel::<AppState>();

    if settings.config.tui {
        thread::spawn(move || {
                run_npag(engine,
                    ranges,
                    theta,
                    &scenarios,
                    c,
                    tx,
                    &settings
                );
            }
        );
        start_ui(rx)?;

    } else {
        run_npag(engine,
            ranges,
            theta,
            &scenarios,
            c,
            tx,
            &settings
        );

    }

    
    
    Ok(())
}

fn run_npag<S>(
    sim_eng: Engine<S>,
    ranges: Vec<(f64,f64)>,
    theta: Array2<f64>,
    scenarios: &Vec<Scenario>,
    c: (f64,f64,f64,f64),
    tx: UnboundedSender<AppState>,
    settings: &Data
)
where
    S: Simulate + std::marker::Sync
{
    let (theta, _w, _objf, _cycle, _converged) = 
        npag(sim_eng,ranges,theta,scenarios,c,tx,settings);
    

    if let Some(theta_path) =  &settings.paths.posterior_dist {
        let file = File::create(theta_path).unwrap();
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
        writer.serialize_array2(&theta).unwrap();
    } 

}

fn setup_log(settings: &Data){
    if let Some(log_path) = &settings.paths.log_out {
        if fs::remove_file(log_path).is_ok(){};
        let logfile = FileAppender::builder()
            .encoder(Box::new(PatternEncoder::new("{l} - {m}\n")))
            .build(log_path).unwrap();

        let config = Config::builder()
            .appender(Appender::builder().build("logfile", Box::new(logfile)))
            .build(Root::builder()
            .appender("logfile")
            .build(LevelFilter::Info)).unwrap();

        log4rs::init_config(config).unwrap();
    };
}