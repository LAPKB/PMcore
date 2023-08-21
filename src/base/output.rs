use crate::prelude::datafile::Scenario;
use crate::prelude::predict::{post_predictions, sim_obs, Engine, Predict};
use crate::prelude::settings::run::Data;
use csv::WriterBuilder;
use ndarray::parallel::prelude::*;
use ndarray::{Array, Array1, Array2, Axis};
use std::error::Error;
use std::fs::File;

pub fn write_theta(
    theta: &Array2<f64>,
    w: &Array1<f64>,
    settings: &Data,
) -> Result<(), Box<dyn Error>> {
    let file = File::create("theta.csv")?;
    let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

    // Create the headers
    let mut theta_header: Vec<String> = settings
        .parsed
        .random
        .iter()
        .map(|(name, _)| name.clone())
        .collect();
    theta_header.push("prob".to_string());

    writer.write_record(&theta_header)?;

    // Combine the theta and w arrays and write them to the CSV
    for (theta_row, &w_val) in theta.outer_iter().zip(w.iter()) {
        let mut row: Vec<String> = theta_row.iter().map(|&val| val.to_string()).collect();
        row.push(w_val.to_string());
        writer.write_record(&row)?;
    }

    writer.flush()?;

    Ok(())
}

pub fn write_posterior(
    psi: &Array2<f64>,
    w: &Array1<f64>,
    settings: &Data,
    theta: &Array2<f64>,
    scenarios: &Vec<Scenario>,
) -> Result<(), Box<dyn Error>> {
    // Assuming you want to propagate the csv errors

    let post_file = File::create("posterior.csv").unwrap();
    let mut post_writer = WriterBuilder::new()
        .has_headers(false)
        .from_writer(post_file);

    let posterior = posterior(psi, w);

    post_writer.write_field("id")?;
    post_writer.write_field("point")?;
    let parameter_names = &settings.computed.random.names;
    for i in 0..theta.ncols() {
        let param_name = parameter_names.get(i).unwrap();
        post_writer.write_field(param_name)?;
    }
    post_writer.write_field("prob")?;
    post_writer.write_record(None::<&[u8]>)?;

    for (sub, row) in posterior.axis_iter(Axis(0)).enumerate() {
        for (spp, elem) in row.axis_iter(Axis(0)).enumerate() {
            post_writer.write_field(&scenarios.get(sub).unwrap().id)?;
            post_writer.write_field(format!("{}", spp))?;
            for param in theta.row(spp) {
                post_writer.write_field(&format!("{param}"))?;
            }
            post_writer.write_field(&format!("{elem:.10}"))?;
            post_writer.write_record(None::<&[u8]>)?;
        }
    }

    Ok(())
}

pub fn write_pred<S>(
    theta: &Array2<f64>,
    w: &Array1<f64>,
    psi: &Array2<f64>,
    sim_eng: Engine<S>,
    scenarios: &Vec<Scenario>,
    cache: bool,
) -> Result<(), Box<dyn Error>>
where
    S: Predict + std::marker::Sync + std::marker::Send + 'static,
{
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
    Ok(for (id, scenario) in scenarios.iter().enumerate() {
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
    })
}

pub fn write_obs(scenarios: &Vec<Scenario>) -> Result<(), Box<dyn Error>> {
    // Create the file
    let output_file = File::create("obs.csv")?;
    let mut csv_writer = WriterBuilder::new()
        .has_headers(false)
        .from_writer(output_file);

    // Write the header
    csv_writer.write_record(&["id", "time", "obs", "outeq"])?;

    // Write the observations
    for scenario in scenarios {
        for (observation, time) in scenario.obs.iter().zip(&scenario.obs_times) {
            csv_writer.write_record(&[
                scenario.id.to_string(),
                time.to_string(),
                observation.to_string(),
                "1".to_string(),
            ])?;
        }
    }

    // Flush the writer to ensure all data is written to file and handle the result
    csv_writer.flush()?;
    Ok(())
}

// Cycles
pub struct CycleWriter {
    writer: csv::Writer<File>,
}

impl CycleWriter {
    pub fn new(file_path: &str, parameter_names: Vec<String>) -> CycleWriter {
        let file = File::create(file_path).unwrap();
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);

        // Write headers
        writer.write_field("cycle").unwrap();
        writer.write_field("neg2ll").unwrap();
        writer.write_field("gamlam").unwrap();
        writer.write_field("nspp").unwrap();

        for param_name in &parameter_names {
            writer.write_field(format!("{}.mean", param_name)).unwrap();
            writer
                .write_field(format!("{}.median", param_name))
                .unwrap();
            writer.write_field(format!("{}.sd", param_name)).unwrap();
        }

        writer.write_record(None::<&[u8]>).unwrap();

        CycleWriter { writer }
    }

    pub fn write(&mut self, cycle: usize, objf: f64, gamma: f64, theta: &Array2<f64>) {
        self.writer.write_field(format!("{}", cycle)).unwrap();
        self.writer.write_field(format!("{}", -2. * objf)).unwrap();
        self.writer.write_field(format!("{}", gamma)).unwrap();
        self.writer
            .write_field(format!("{}", theta.nrows()))
            .unwrap();

        for param in theta.axis_iter(Axis(1)) {
            self.writer
                .write_field(format!("{}", param.mean().unwrap()))
                .unwrap();
        }

        for param in theta.axis_iter(Axis(1)) {
            self.writer
                .write_field(format!("{}", median(param.to_owned().to_vec())))
                .unwrap();
        }

        for param in theta.axis_iter(Axis(1)) {
            self.writer
                .write_field(format!("{}", param.std(1.)))
                .unwrap();
        }

        self.writer.write_record(None::<&[u8]>).unwrap();
    }

    pub fn flush(&mut self) {
        self.writer.flush().unwrap();
    }
}

pub fn posterior(psi: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
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

pub fn median(data: Vec<f64>) -> f64 {
    let size = data.len();
    match size {
        even if even % 2 == 0 => {
            let fst = data.get(even / 2 - 1).unwrap();
            let snd = data.get(even / 2).unwrap();
            (fst + snd) / 2.0
        }
        odd => *data.get(odd / 2_usize).unwrap(),
    }
}

pub fn population_mean_median(theta: &Array2<f64>, w: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
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

pub fn posterior_mean_median(
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
        psi_norm.push_row(row_norm.view()).unwrap();
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
