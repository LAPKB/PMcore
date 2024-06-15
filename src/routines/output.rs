use crate::prelude::*;
use csv::WriterBuilder;
use ndarray::parallel::prelude::*;
use ndarray::{Array, Array1, Array2, Axis};
use pharmsol::prelude::data::*;
use pharmsol::prelude::simulator::Equation;
use settings::Settings;
use std::fs::File;

/// Defines the result objects from an NPAG run
/// An [NPResult] contains the necessary information to generate predictions and summary statistics
#[derive(Debug)]
pub struct NPResult {
    pub data: Data,
    pub theta: Array2<f64>,
    pub psi: Array2<f64>,
    pub w: Array1<f64>,
    pub objf: f64,
    pub cycles: usize,
    pub converged: bool,
    pub par_names: Vec<String>,
    pub settings: Settings,
}

impl NPResult {
    /// Create a new NPResult object
    pub fn new(
        data: Data,
        theta: Array2<f64>,
        psi: Array2<f64>,
        w: Array1<f64>,
        objf: f64,
        cycles: usize,
        converged: bool,
        settings: Settings,
    ) -> Self {
        // TODO: Add support for fixed and constant parameters

        let par_names = settings.random.names();

        Self {
            data,
            theta,
            psi,
            w,
            objf,
            cycles,
            converged,
            par_names,
            settings,
        }
    }

    pub fn write_outputs(&self, write: bool, equation: &Equation, idelta: f64, tad: f64) {
        if write {
            self.write_theta();
            self.write_posterior();
            self.write_obs();
            self.write_pred(equation, idelta, tad);
        }
    }

    /// Writes theta, which containts the population support points and their associated probabilities
    /// Each row is one support point, the last column being probability
    pub fn write_theta(&self) {
        tracing::info!("Writing final parameter distribution...");
        let result = (|| {
            let theta: Array2<f64> = self.theta.clone();
            let w: Array1<f64> = self.w.clone();

            let file = create_output_file(&self.settings, "theta.csv")?;
            let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

            // Create the headers
            let mut theta_header = self.par_names.clone();
            theta_header.push("prob".to_string());
            writer.write_record(&theta_header)?;

            // Write contents
            for (theta_row, &w_val) in theta.outer_iter().zip(w.iter()) {
                let mut row: Vec<String> = theta_row.iter().map(|&val| val.to_string()).collect();
                row.push(w_val.to_string());
                writer.write_record(&row)?;
            }
            writer.flush()
        })();

        if let Err(e) = result {
            tracing::error!("Error while writing theta: {}", e);
        }
    }

    /// Writes the posterior support points for each individual
    pub fn write_posterior(&self) {
        tracing::info!("Writing posterior parameter probabilities...");
        let result = (|| {
            let theta: Array2<f64> = self.theta.clone();
            let w: Array1<f64> = self.w.clone();
            let psi: Array2<f64> = self.psi.clone();
            let par_names: Vec<String> = self.par_names.clone();
            //let subjects = self.subjects.clone();

            let posterior = posterior(&psi, &w);

            // Create the output folder if it doesn't exist
            let file = create_output_file(&self.settings, "posterior.csv")?;
            let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

            // Create the headers
            writer.write_field("id")?;
            writer.write_field("point")?;
            for i in 0..theta.ncols() {
                let param_name = par_names.get(i).unwrap();
                writer.write_field(param_name)?;
            }
            writer.write_field("prob")?;
            writer.write_record(None::<&[u8]>)?;

            // Write contents
            let subjects = self.data.get_subjects();
            for (sub, row) in posterior.axis_iter(Axis(0)).enumerate() {
                for (spp, elem) in row.axis_iter(Axis(0)).enumerate() {
                    writer.write_field(&subjects.get(sub).unwrap().id())?;
                    writer.write_field(format!("{}", spp))?;
                    for param in theta.row(spp) {
                        writer.write_field(&format!("{param}"))?;
                    }
                    writer.write_field(&format!("{elem:.10}"))?;
                    writer.write_record(None::<&[u8]>)?;
                }
            }
            writer.flush()
        })();

        if let Err(e) = result {
            tracing::error!("Error while writing posterior: {}", e);
        }
    }

    /// Write the observations, which is the reformatted input data
    pub fn write_obs(&self) {
        tracing::info!("Writing observations...");
        let file = create_output_file(&self.settings, "obs.csv").unwrap();
        write_pmetrics_observations(&self.data, &file)
    }

    /// Writes the predictions
    pub fn write_pred(&self, equation: &Equation, idelta: f64, tad: f64) {
        tracing::info!("Writing predictions...");
        let data = self.data.expand(idelta, tad);
        // println!("{:?}", data);

        let theta: Array2<f64> = self.theta.clone();
        let w: Array1<f64> = self.w.clone();
        let psi: Array2<f64> = self.psi.clone();

        let (post_mean, post_median) = posterior_mean_median(&theta, &psi, &w);
        let (pop_mean, pop_median) = population_mean_median(&theta, &w);

        let subjects = data.get_subjects();
        if subjects.len() != post_mean.nrows() {
            panic!("Number of subjects and number of posterior means do not match");
        }

        let file = create_output_file(&self.settings, "pred.csv");
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(file.unwrap());

        // Create the headers
        writer
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

        for (i, subject) in subjects.iter().enumerate() {
            // Population predictions
            let pop_mean_pred = equation
                .simulate_subject(subject, &pop_mean.to_vec())
                .get_predictions()
                .clone();
            let pop_median_pred = equation
                .simulate_subject(subject, &pop_median.to_vec())
                .get_predictions()
                .clone();

            // Posterior predictions
            let post_mean_spp: Vec<f64> = post_mean.row(i).to_vec();
            let post_mean_pred = equation
                .simulate_subject(subject, &post_mean_spp)
                .get_predictions()
                .clone();
            let post_median_spp: Vec<f64> = post_median.row(i).to_vec();
            let post_median_pred = equation
                .simulate_subject(subject, &post_median_spp)
                .get_predictions()
                .clone();

            pop_mean_pred
                .iter()
                .zip(pop_median_pred.iter())
                .zip(post_mean_pred.iter())
                .zip(post_median_pred.iter())
                .for_each(|(((pop_mean, pop_median), post_mean), post_median)| {
                    writer
                        .write_record([
                            subject.id(),
                            &format!("{:.3}", pop_mean.time()),
                            &format!("{}", pop_mean.outeq()),
                            &format!("{:.4}", pop_mean.prediction()),
                            &format!("{:.4}", pop_median.prediction()),
                            &format!("{:.4}", post_mean.prediction()),
                            &format!("{:.4}", post_median.prediction()),
                        ])
                        .unwrap();
                });
        }
    }
}
#[derive(Debug)]
pub struct CycleLog {
    pub cycles: Vec<NPCycle>,
    cycle_writer: CycleWriter,
}
impl CycleLog {
    pub fn new(settings: &Settings) -> Self {
        let cycle_writer = CycleWriter::new(settings);
        Self {
            cycles: Vec::new(),
            cycle_writer,
        }
    }
    pub fn push_and_write(&mut self, npcycle: NPCycle, write_ouput: bool) {
        if write_ouput {
            self.cycle_writer.write(
                npcycle.cycle,
                npcycle.converged,
                npcycle.objf,
                npcycle.gamlam,
                &npcycle.theta,
            );
            self.cycle_writer.flush();
        }
        self.cycles.push(npcycle);
    }
}

/// Defines the result objects from a run
/// An [NPCycle] contains summary of a cycle
/// It holds the following information:
/// - `cycle`: The cycle number
/// - `objf`: The objective function value
/// - `gamlam`: The assay noise parameter, either gamma or lambda
/// - `theta`: The support points and their associated probabilities
/// - `nspp`: The number of support points
/// - `delta_objf`: The change in objective function value from last cycle
/// - `converged`: Whether the algorithm has reached convergence
#[derive(Debug, Clone)]
pub struct NPCycle {
    pub cycle: usize,
    pub objf: f64,
    pub gamlam: f64,
    pub theta: Array2<f64>,
    pub nspp: usize,
    pub delta_objf: f64,
    pub converged: bool,
}
impl NPCycle {
    pub fn new() -> Self {
        Self {
            cycle: 0,
            objf: 0.0,
            gamlam: 0.0,
            theta: Array2::default((0, 0)),
            nspp: 0,
            delta_objf: 0.0,
            converged: false,
        }
    }
}
impl Default for NPCycle {
    fn default() -> Self {
        Self::new()
    }
}

// Cycles
#[derive(Debug)]
pub struct CycleWriter {
    writer: Option<csv::Writer<File>>,
}

impl CycleWriter {
    pub fn new(settings: &Settings) -> CycleWriter {
        let writer = if settings.config.output {
            let file = create_output_file(settings, "cycles.csv").unwrap();
            let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);

            // Write headers
            writer.write_field("cycle").unwrap();
            writer.write_field("converged").unwrap();
            writer.write_field("neg2ll").unwrap();
            writer.write_field("gamlam").unwrap();
            writer.write_field("nspp").unwrap();

            let parameter_names = settings.random.names();
            for param_name in &parameter_names {
                writer.write_field(format!("{}.mean", param_name)).unwrap();
                writer
                    .write_field(format!("{}.median", param_name))
                    .unwrap();
                writer.write_field(format!("{}.sd", param_name)).unwrap();
            }

            writer.write_record(None::<&[u8]>).unwrap();
            Some(writer)
        } else {
            None
        };

        CycleWriter { writer }
    }

    pub fn write(
        &mut self,
        cycle: usize,
        converged: bool,
        objf: f64,
        gamma: f64,
        theta: &Array2<f64>,
    ) {
        if let Some(writer) = &mut self.writer {
            writer.write_field(format!("{}", cycle)).unwrap();
            writer.write_field(format!("{}", converged)).unwrap();
            writer.write_field(format!("{}", objf)).unwrap();
            writer.write_field(format!("{}", gamma)).unwrap();
            writer.write_field(format!("{}", theta.nrows())).unwrap();

            for param in theta.axis_iter(Axis(1)) {
                writer
                    .write_field(format!("{}", param.mean().unwrap()))
                    .unwrap();
            }

            for param in theta.axis_iter(Axis(1)) {
                writer
                    .write_field(format!("{}", median(param.to_owned().to_vec())))
                    .unwrap();
            }

            for param in theta.axis_iter(Axis(1)) {
                writer.write_field(format!("{}", param.std(1.))).unwrap();
            }

            writer.write_record(None::<&[u8]>).unwrap();
        }
    }

    pub fn flush(&mut self) {
        if let Some(writer) = &mut self.writer {
            writer.flush().unwrap(); // Handle errors appropriately
        }
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

fn weighted_median(data: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    // Ensure the parameters and weights vectors have the same length
    assert_eq!(
        data.len(),
        weights.len(),
        "The length of parameters and weights must be the same"
    );
    // Handle edge case where all parameters are the same
    if data.iter().all(|&x| x == data[0]) {
        return data[0];
    }

    // // Handle the edge case where there is only one parameter
    // if data.len() == 1 {
    //     return data[0];
    // }
    let mut tup: Vec<(f64, f64)> = Vec::new();

    for (ti, wi) in data.iter().zip(weights) {
        tup.push((*ti, *wi));
    }

    tup.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    if tup.first().unwrap().1 >= 0.5 {
        tup.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    }

    let mut wacc: Vec<f64> = Vec::new();
    let mut widx: usize = 0;

    for (i, (ti, wi)) in tup.iter().enumerate() {
        let acc = wi + wacc.last().unwrap_or(&0.0);
        wacc.push(acc);

        if acc > 0.5 {
            widx = i;
            break;
        } else if acc == 0.5 {
            return *ti;
        }
    }

    let acc2 = wacc.pop().unwrap();
    let acc1 = wacc.pop().unwrap();
    let par2 = tup.get(widx).unwrap().0;
    let par1 = tup.get(widx - 1).unwrap().0;
    let slope = (par2 - par1) / (acc2 - acc1);
    par1 + slope * (0.5 - acc1)
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
        let mut params = vec![];
        let mut weights = vec![];
        for (ti, wi) in ct.iter().zip(w) {
            params.push(*ti);
            weights.push(*wi);
        }

        *mdn = weighted_median(&Array::from(params), &Array::from(weights));
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
            let median = weighted_median(&pars.to_owned(), &probs.to_owned());
            post_median.push(median);
        }

        mean.push_row(Array::from(post_mean.clone()).view())
            .unwrap();
        median
            .push_row(Array::from(post_median.clone()).view())
            .unwrap();
    }

    (mean, median)
}

pub fn create_output_file(settings: &Settings, file_name: &str) -> std::io::Result<File> {
    let output_folder = settings
        .paths
        .output_folder
        .as_ref()
        .ok_or(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Output folder not specified in settings",
        ))?;

    let path = std::path::Path::new(output_folder);

    // Attempt to create the output directory (does nothing if already exists)
    std::fs::create_dir_all(path)?;

    // Create and open the file, returning the File handle
    let file_path = path.join(file_name);
    File::create(file_path)
}

pub fn write_pmetrics_observations(data: &Data, file: &std::fs::File) {
    let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

    writer
        .write_record(&["id", "block", "time", "out", "outeq"])
        .unwrap();
    for subject in data.get_subjects() {
        for occasion in subject.occasions() {
            for event in occasion.get_events(None, None, false) {
                match event {
                    Event::Observation(obs) => {
                        // Write each field individually
                        writer
                            .write_record(&[
                                &subject.id(),
                                &occasion.index().to_string(),
                                &obs.time().to_string(),
                                &obs.value().to_string(),
                                &obs.outeq().to_string(),
                            ])
                            .unwrap();
                    }
                    _ => {}
                }
            }
        }
    }
}
