use self::datafile::Scenario;
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
use ndarray::parallel::prelude::*;
use ndarray::{Array, Array1, Array2, Axis};
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
    if let Some(exclude) = &settings.config.exclude {
        for val in exclude {
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
    let (theta, psi, w, _objf, _cycle, _converged) =
        npag(&sim_eng, ranges, theta, scenarios, c, tx, settings);

    if let Some(output) = &settings.config.pmetrics_outputs {
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

            // posterior.csv
            let posterior = posterior(&psi, &w);
            let file = File::create("posterior.csv").unwrap();
            let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
            writer.serialize_array2(&posterior).unwrap();

            // //pred.csv
            // let pred = sim_obs(&sim_eng, scenarios, &theta);
            let (_pop_mean, _pop_median) = population_mean_median(&theta, &w);

            // For debugging
            let posts = posterior_mean_median(&theta, &psi, &w);

            //obs.csv
            let obs_file = File::create("obs.csv").unwrap();
            let mut obs_writer = WriterBuilder::new()
                .has_headers(false)
                .from_writer(obs_file);
            obs_writer
                .write_record(["sub_num", "time", "obs", "outeq"])
                .unwrap();
            for (id, scenario) in scenarios.iter().enumerate() {
                let observations = scenario.obs_flat.clone();
                let time = scenario.time_flat.clone();

                for (obs, t) in observations.into_iter().zip(time) {
                    obs_writer
                        .write_record(&[
                            id.to_string(),
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

fn posterior(psi: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    let py = psi.dot(w);
    let mut post: Array2<f64> = Array2::zeros((psi.nrows(), psi.ncols()));
    post.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            row.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(j, mut element)| {
                    let elem = psi.get((i, j)).unwrap() * w.get(j).unwrap() / py.get(i).unwrap();
                    element.fill(elem);
                });
        });
    post
}

fn population_mean_median(theta: &Array2<f64>, w: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let mut mean = Array1::zeros(theta.ncols());
    let mut median = Array1::zeros(theta.ncols());

    for (i, (mn, mdn)) in mean.iter_mut().zip(&mut median).enumerate() {
        // Calculate the weighted mean
        let col = theta.column(i).to_owned() * w.to_owned();
        *mn = col.sum();

        // Calculate the median
        let ct = theta.column(i);
        let mut tup: Vec<(f64, f64)> = Vec::new();
        for (ti, wi) in ct.iter().zip(w) {
            tup.push((*ti, *wi));
        }

        tup.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());

        let mut wacc: Vec<f64> = Vec::new();
        let mut widx: usize = 0;

        for (i, (_, wi)) in tup.iter().enumerate() {
            let acc = wi + wacc.last().unwrap_or(&0.0);
            wacc.push(acc);

            if acc > 0.5 {
                widx = i;
                break;
            }
        }

        let acc2 = wacc.pop().unwrap();
        let acc1 = wacc.pop().unwrap();
        let par2 = tup.get(widx).unwrap().0;
        let par1 = tup.get(widx - 1).unwrap().0;
        let slope = (par2 - par1) / (acc2 - acc1);

        *mdn = par1 + slope * (0.5 - acc1);
    }

    (mean, median)
}

fn posterior_mean_median(
    theta: &Array2<f64>,
    psi: &Array2<f64>,
    w: &Array1<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let mut mean = Array2::zeros((0, theta.ncols()));
    let mut median = Array2::zeros((0, theta.ncols()));

    // Normalize psi to get probabilities of each spp for each id
    let mut psi_norm: Array2<f64> = Array2::zeros((0, psi.ncols()));
    for row in psi.axis_iter(Axis(0)) {
        let row_w = row.to_owned() * w.to_owned();
        let row_sum = row_w.sum();
        let row_norm = &row_w / row_sum;
        psi_norm.push_row(row_norm.view());
    }

    // Transpose normalized psi to get ID (col) by prob (row)
    let psi_norm_transposed = psi_norm.t();

    // For each subject..
    for probs in psi_norm_transposed.axis_iter(Axis(1)) {
        let mut post_mean: Vec<f64> = Vec::new();
        let mut post_median: Vec<f64> = Vec::new();

        // For each parameter
        for pars in theta.axis_iter(Axis(1)) {
            // Calculate the mean
            let weighted_par = &probs * &pars;
            let the_mean = weighted_par.sum();
            post_mean.push(the_mean);

            // Calculate the median
            let mut tup: Vec<(f64, f64)> = Vec::new();

            for (ti, wi) in pars.iter().zip(probs) {
                tup.push((*ti, *wi));
            }

            tup.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());

            if tup.first().unwrap().1 >= 0.5 {
                tup.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());
            }

            let mut wacc: Vec<f64> = Vec::new();
            let mut widx: usize = 0;

            for (i, (_, wi)) in tup.iter().enumerate() {
                let acc = wi + wacc.last().unwrap_or(&0.0);
                wacc.push(acc);

                if acc > 0.5 {
                    widx = i;
                    break;
                }
            }

            let acc2 = wacc.pop().unwrap();
            let acc1 = wacc.pop().unwrap();
            let par2 = tup.get(widx).unwrap().0;
            let par1 = tup.get(widx - 1).unwrap().0;
            let slope = (par2 - par1) / (acc2 - acc1);
            let the_median = par1 + slope * (0.5 - acc1);
            post_median.push(the_median);
        }

        mean.push_row(Array::from(post_mean.clone()).view())
            .unwrap();
        median
            .push_row(Array::from(post_median.clone()).view())
            .unwrap();
    }

    (mean, median)
}
