use crate::algorithms::Status;
use crate::prelude::*;
use crate::routines::output::cycles::CycleLog;
use crate::routines::output::predictions::NPPredictions;
use crate::routines::settings::Settings;
use crate::structs::psi::Psi;
use crate::structs::theta::Theta;
use crate::structs::weights::Weights;
use anyhow::{bail, Context, Result};
use csv::WriterBuilder;
use faer::linalg::zip::IntoView;
use faer_ext::IntoNdarray;
use ndarray::{Array, Array1, Array2, Axis};
use pharmsol::prelude::data::*;
use pharmsol::prelude::simulator::Equation;
use serde::Serialize;
use std::fs::{create_dir_all, File, OpenOptions};
use std::path::{Path, PathBuf};

pub mod cycles;
pub mod posterior;
pub mod predictions;

use posterior::posterior;

/// Defines the result objects from an NPAG run
/// An [NPResult] contains the necessary information to generate predictions and summary statistics
#[derive(Debug, Serialize)]
pub struct NPResult<E: Equation> {
    #[serde(skip)]
    equation: E,
    data: Data,
    theta: Theta,
    psi: Psi,
    w: Weights,
    objf: f64,
    cycles: usize,
    status: Status,
    par_names: Vec<String>,
    settings: Settings,
    cyclelog: CycleLog,
}

#[allow(clippy::too_many_arguments)]
impl<E: Equation> NPResult<E> {
    /// Create a new NPResult object
    pub fn new(
        equation: E,
        data: Data,
        theta: Theta,
        psi: Psi,
        w: Weights,
        objf: f64,
        cycles: usize,
        status: Status,
        settings: Settings,
        cyclelog: CycleLog,
    ) -> Self {
        // TODO: Add support for fixed and constant parameters

        let par_names = settings.parameters().names();

        Self {
            equation,
            data,
            theta,
            psi,
            w,
            objf,
            cycles,
            status,
            par_names,
            settings,
            cyclelog,
        }
    }

    pub fn cycles(&self) -> usize {
        self.cycles
    }

    pub fn objf(&self) -> f64 {
        self.objf
    }

    pub fn converged(&self) -> bool {
        self.status == Status::Converged
    }

    pub fn get_theta(&self) -> &Theta {
        &self.theta
    }

    /// Get the [Psi] structure
    pub fn psi(&self) -> &Psi {
        &self.psi
    }

    /// Get the weights (probabilities) of the support points
    pub fn weights(&self) -> &Weights {
        &self.w
    }

    pub fn write_outputs(&self) -> Result<()> {
        if self.settings.output().write {
            tracing::debug!("Writing outputs to {:?}", self.settings.output().path);
            self.settings.write()?;
            let idelta: f64 = self.settings.predictions().idelta;
            let tad = self.settings.predictions().tad;
            self.cyclelog.write(&self.settings)?;
            self.write_theta().context("Failed to write theta")?;
            self.write_covs().context("Failed to write covariates")?;
            self.write_predictions(idelta, tad)
                .context("Failed to write predictions")?;
            self.write_posterior()
                .context("Failed to write posterior")?;
        }
        Ok(())
    }

    /// Writes the observations and predictions to a single file
    pub fn write_obspred(&self) -> Result<()> {
        tracing::debug!("Writing observations and predictions...");

        #[derive(Debug, Clone, Serialize)]
        struct Row {
            id: String,
            time: f64,
            outeq: usize,
            block: usize,
            obs: Option<f64>,
            pop_mean: f64,
            pop_median: f64,
            post_mean: f64,
            post_median: f64,
        }

        let theta: Array2<f64> = self
            .theta
            .matrix()
            .clone()
            .as_mut()
            .into_ndarray()
            .to_owned();
        let w: Array1<f64> = self
            .w
            .weights()
            .clone()
            .into_view()
            .iter()
            .cloned()
            .collect();
        let psi: Array2<f64> = self.psi.matrix().as_ref().into_ndarray().to_owned();

        let (post_mean, post_median) = posterior_mean_median(&theta, &psi, &w)
            .context("Failed to calculate posterior mean and median")?;

        let (pop_mean, pop_median) = population_mean_median(&theta, &w)
            .context("Failed to calculate posterior mean and median")?;

        let subjects = self.data.subjects();
        if subjects.len() != post_mean.nrows() {
            bail!(
                "Number of subjects: {} and number of posterior means: {} do not match",
                subjects.len(),
                post_mean.nrows()
            );
        }

        let outputfile = OutputFile::new(&self.settings.output().path, "op.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(&outputfile.file);

        for (i, subject) in subjects.iter().enumerate() {
            for occasion in subject.occasions() {
                let id = subject.id();
                let occ = occasion.index();

                let subject = Subject::from_occasions(id.clone(), vec![occasion.clone()]);

                // Population predictions
                let pop_mean_pred = self
                    .equation
                    .simulate_subject(&subject, &pop_mean.to_vec(), None)?
                    .0
                    .get_predictions()
                    .clone();

                let pop_median_pred = self
                    .equation
                    .simulate_subject(&subject, &pop_median.to_vec(), None)?
                    .0
                    .get_predictions()
                    .clone();

                // Posterior predictions
                let post_mean_spp: Vec<f64> = post_mean.row(i).to_vec();
                let post_mean_pred = self
                    .equation
                    .simulate_subject(&subject, &post_mean_spp, None)?
                    .0
                    .get_predictions()
                    .clone();
                let post_median_spp: Vec<f64> = post_median.row(i).to_vec();
                let post_median_pred = self
                    .equation
                    .simulate_subject(&subject, &post_median_spp, None)?
                    .0
                    .get_predictions()
                    .clone();
                assert_eq!(
                    pop_mean_pred.len(),
                    pop_median_pred.len(),
                    "The number of predictions do not match (pop_mean vs pop_median)"
                );

                assert_eq!(
                    post_mean_pred.len(),
                    post_median_pred.len(),
                    "The number of predictions do not match (post_mean vs post_median)"
                );

                assert_eq!(
                    pop_mean_pred.len(),
                    post_mean_pred.len(),
                    "The number of predictions do not match (pop_mean vs post_mean)"
                );

                for (((pop_mean_pred, pop_median_pred), post_mean_pred), post_median_pred) in
                    pop_mean_pred
                        .iter()
                        .zip(pop_median_pred.iter())
                        .zip(post_mean_pred.iter())
                        .zip(post_median_pred.iter())
                {
                    let row = Row {
                        id: id.clone(),
                        time: pop_mean_pred.time(),
                        outeq: pop_mean_pred.outeq(),
                        block: occ,
                        obs: pop_mean_pred.observation(),
                        pop_mean: pop_mean_pred.prediction(),
                        pop_median: pop_median_pred.prediction(),
                        post_mean: post_mean_pred.prediction(),
                        post_median: post_median_pred.prediction(),
                    };
                    writer.serialize(row)?;
                }
            }
        }
        writer.flush()?;
        tracing::debug!(
            "Observations with predictions written to {:?}",
            &outputfile.relative_path()
        );
        Ok(())
    }

    /// Writes theta, which contains the population support points and their associated probabilities
    /// Each row is one support point, the last column being probability
    pub fn write_theta(&self) -> Result<()> {
        tracing::debug!("Writing population parameter distribution...");

        let theta = &self.theta;
        let w: Vec<f64> = self
            .w
            .weights()
            .clone()
            .into_view()
            .iter()
            .cloned()
            .collect();

        if w.len() != theta.matrix().nrows() {
            bail!(
                "Number of weights ({}) and number of support points ({}) do not match.",
                w.len(),
                theta.matrix().nrows()
            );
        }

        let outputfile = OutputFile::new(&self.settings.output().path, "theta.csv")
            .context("Failed to create output file for theta")?;

        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(&outputfile.file);

        // Create the headers
        let mut theta_header = self.par_names.clone();
        theta_header.push("prob".to_string());
        writer.write_record(&theta_header)?;

        // Write contents
        for (theta_row, &w_val) in theta.matrix().row_iter().zip(w.iter()) {
            let mut row: Vec<String> = theta_row.iter().map(|&val| val.to_string()).collect();
            row.push(w_val.to_string());
            writer.write_record(&row)?;
        }
        writer.flush()?;
        tracing::debug!(
            "Population parameter distribution written to {:?}",
            &outputfile.relative_path()
        );
        Ok(())
    }

    /// Writes the posterior support points for each individual
    pub fn write_posterior(&self) -> Result<()> {
        tracing::debug!("Writing posterior parameter probabilities...");
        let theta = &self.theta;
        let w = &self.w;
        let psi = &self.psi;

        // Calculate the posterior probabilities
        let posterior = posterior(psi, w)?;

        // Create the output folder if it doesn't exist
        let outputfile = match OutputFile::new(&self.settings.output().path, "posterior.csv") {
            Ok(of) => of,
            Err(e) => {
                tracing::error!("Failed to create output file: {}", e);
                return Err(e.context("Failed to create output file"));
            }
        };

        // Create a new writer
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(&outputfile.file);

        // Create the headers
        writer.write_field("id")?;
        writer.write_field("point")?;
        theta.param_names().iter().for_each(|name| {
            writer.write_field(name).unwrap();
        });
        writer.write_field("prob")?;
        writer.write_record(None::<&[u8]>)?;

        // Write contents
        let subjects = self.data.subjects();
        posterior
            .matrix()
            .row_iter()
            .enumerate()
            .for_each(|(i, row)| {
                let subject = subjects.get(i).unwrap();
                let id = subject.id();

                row.iter().enumerate().for_each(|(spp, prob)| {
                    writer.write_field(id.clone()).unwrap();
                    writer.write_field(spp.to_string()).unwrap();

                    theta.matrix().row(spp).iter().for_each(|val| {
                        writer.write_field(val.to_string()).unwrap();
                    });

                    writer.write_field(prob.to_string()).unwrap();
                    writer.write_record(None::<&[u8]>).unwrap();
                });
            });

        writer.flush()?;
        tracing::debug!(
            "Posterior parameters written to {:?}",
            &outputfile.relative_path()
        );

        Ok(())
    }

    /// Writes the predictions
    pub fn write_predictions(&self, idelta: f64, tad: f64) -> Result<()> {
        tracing::debug!("Writing predictions...");

        let posterior = posterior(&self.psi, &self.w)?;

        // Calculate the predictions
        let predictions = NPPredictions::calculate(
            &self.equation,
            &self.data,
            self.theta.clone(),
            &self.w,
            &posterior,
            idelta,
            tad,
        )?;

        // Write (full) predictions to pred.csv
        let outputfile_pred = OutputFile::new(&self.settings.output().path, "pred.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(&outputfile_pred.file);

        // Write each prediction row
        for row in predictions.predictions() {
            writer.serialize(row)?;
        }

        writer.flush()?;
        tracing::debug!(
            "Predictions written to {:?}",
            &outputfile_pred.relative_path()
        );

        Ok(())
    }

    /// Writes the covariates
    pub fn write_covs(&self) -> Result<()> {
        tracing::debug!("Writing covariates...");
        let outputfile = OutputFile::new(&self.settings.output().path, "covs.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(&outputfile.file);

        // Collect all unique covariate names
        let mut covariate_names = std::collections::HashSet::new();
        for subject in self.data.subjects() {
            for occasion in subject.occasions() {
                let cov = occasion.covariates();
                let covmap = cov.covariates();
                for cov_name in covmap.keys() {
                    covariate_names.insert(cov_name.clone());
                }
            }
        }
        let mut covariate_names: Vec<String> = covariate_names.into_iter().collect();
        covariate_names.sort(); // Ensure consistent order

        // Write the header row: id, time, block, covariate names
        let mut headers = vec!["id", "time", "block"];
        headers.extend(covariate_names.iter().map(|s| s.as_str()));
        writer.write_record(&headers)?;

        // Write the data rows
        for subject in self.data.subjects() {
            for occasion in subject.occasions() {
                let cov = occasion.covariates();
                let covmap = cov.covariates();

                for event in occasion.iter() {
                    let time = match event {
                        Event::Bolus(bolus) => bolus.time(),
                        Event::Infusion(infusion) => infusion.time(),
                        Event::Observation(observation) => observation.time(),
                    };

                    let mut row: Vec<String> = Vec::new();
                    row.push(subject.id().clone());
                    row.push(time.to_string());
                    row.push(occasion.index().to_string());

                    // Add covariate values to the row
                    for cov_name in &covariate_names {
                        if let Some(cov) = covmap.get(cov_name) {
                            if let Ok(value) = cov.interpolate(time) {
                                row.push(value.to_string());
                            } else {
                                row.push(String::new());
                            }
                        } else {
                            row.push(String::new());
                        }
                    }

                    writer.write_record(&row)?;
                }
            }
        }

        writer.flush()?;
        tracing::debug!("Covariates written to {:?}", &outputfile.relative_path());
        Ok(())
    }
}

pub(crate) fn median(data: &[f64]) -> f64 {
    let mut data: Vec<f64> = data.to_vec();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

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

fn weighted_median(data: &[f64], weights: &[f64]) -> f64 {
    // Ensure the data and weights arrays have the same length
    assert_eq!(
        data.len(),
        weights.len(),
        "The length of data and weights must be the same"
    );
    assert!(
        weights.iter().all(|&x| x >= 0.0),
        "Weights must be non-negative, weights: {:?}",
        weights
    );

    // Create a vector of tuples (data, weight)
    let mut weighted_data: Vec<(f64, f64)> = data
        .iter()
        .zip(weights.iter())
        .map(|(&d, &w)| (d, w))
        .collect();

    // Sort the vector by the data values
    weighted_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Calculate the cumulative sum of weights
    let total_weight: f64 = weights.iter().sum();
    let mut cumulative_sum = 0.0;

    for (i, &(_, weight)) in weighted_data.iter().enumerate() {
        cumulative_sum += weight;

        if cumulative_sum == total_weight / 2.0 {
            // If the cumulative sum equals half the total weight, average this value with the next
            if i + 1 < weighted_data.len() {
                return (weighted_data[i].0 + weighted_data[i + 1].0) / 2.0;
            } else {
                return weighted_data[i].0;
            }
        } else if cumulative_sum > total_weight / 2.0 {
            return weighted_data[i].0;
        }
    }

    unreachable!("The function should have returned a value before reaching this point.");
}

pub fn population_mean_median(
    theta: &Array2<f64>,
    w: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let w = if w.is_empty() {
        tracing::warn!("w.len() == 0, setting all weights to 1/n");
        Array1::from_elem(theta.nrows(), 1.0 / theta.nrows() as f64)
    } else {
        w.clone()
    };
    // Check for compatible sizes
    if theta.nrows() != w.len() {
        bail!(
            "Number of parameters and number of weights do not match. Theta: {}, w: {}",
            theta.nrows(),
            w.len()
        );
    }

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
        for (ti, wi) in ct.iter().zip(w.clone()) {
            params.push(*ti);
            weights.push(wi);
        }

        *mdn = weighted_median(&params, &weights);
    }

    Ok((mean, median))
}

pub fn posterior_mean_median(
    theta: &Array2<f64>,
    psi: &Array2<f64>,
    w: &Array1<f64>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let mut mean = Array2::zeros((0, theta.ncols()));
    let mut median = Array2::zeros((0, theta.ncols()));

    let w = if w.is_empty() {
        tracing::warn!("w is empty, setting all weights to 1/n");
        Array1::from_elem(theta.nrows(), 1.0 / theta.nrows() as f64)
    } else {
        w.clone()
    };

    // Check for compatible sizes
    if theta.nrows() != w.len() || theta.nrows() != psi.ncols() || psi.ncols() != w.len() {
        bail!("Number of parameters and number of weights do not match, theta.nrows(): {}, w.len(): {}, psi.ncols(): {}", theta.nrows(), w.len(), psi.ncols());
    }

    // Normalize psi to get probabilities of each spp for each id
    let mut psi_norm: Array2<f64> = Array2::zeros((0, psi.ncols()));
    for (i, row) in psi.axis_iter(Axis(0)).enumerate() {
        let row_w = row.to_owned() * w.to_owned();
        let row_sum = row_w.sum();
        let row_norm = if row_sum == 0.0 {
            tracing::warn!("Sum of row {} of psi is 0.0, setting that row to 1/n", i);
            Array1::from_elem(psi.ncols(), 1.0 / psi.ncols() as f64)
        } else {
            &row_w / row_sum
        };
        psi_norm.push_row(row_norm.view())?;
    }
    if psi_norm.iter().any(|&x| x.is_nan()) {
        dbg!(&psi);
        bail!("NaN values found in psi_norm");
    };

    // Transpose normalized psi to get ID (col) by prob (row)
    // let psi_norm_transposed = psi_norm.t();

    // For each subject..
    for probs in psi_norm.axis_iter(Axis(0)) {
        let mut post_mean: Vec<f64> = Vec::new();
        let mut post_median: Vec<f64> = Vec::new();

        // For each parameter
        for pars in theta.axis_iter(Axis(1)) {
            // Calculate the mean
            let weighted_par = &probs * &pars;
            let the_mean = weighted_par.sum();
            post_mean.push(the_mean);

            // Calculate the median
            let median = weighted_median(&pars.to_vec(), &probs.to_vec());
            post_median.push(median);
        }

        mean.push_row(Array::from(post_mean.clone()).view())?;
        median.push_row(Array::from(post_median.clone()).view())?;
    }

    Ok((mean, median))
}

/// Contains all the necessary information of an output file
#[derive(Debug)]
pub struct OutputFile {
    file: File,
    relative_path: PathBuf,
}

impl OutputFile {
    pub fn new(folder: &str, file_name: &str) -> Result<Self> {
        let relative_path = Path::new(&folder).join(file_name);

        if let Some(parent) = relative_path.parent() {
            create_dir_all(parent)
                .with_context(|| format!("Failed to create directories for {:?}", parent))?;
        }

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&relative_path)
            .with_context(|| format!("Failed to open file: {:?}", relative_path))?;

        Ok(OutputFile {
            file,
            relative_path,
        })
    }

    pub fn file(&self) -> &File {
        &self.file
    }

    pub fn file_owned(self) -> File {
        self.file
    }

    pub fn relative_path(&self) -> &Path {
        &self.relative_path
    }
}

#[cfg(test)]
mod tests {
    use super::median;

    #[test]
    fn test_median_odd() {
        let data = vec![1.0, 3.0, 2.0];
        assert_eq!(median(&data), 2.0);
    }

    #[test]
    fn test_median_even() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&data), 2.5);
    }

    #[test]
    fn test_median_single() {
        let data = vec![42.0];
        assert_eq!(median(&data), 42.0);
    }

    #[test]
    fn test_median_sorted() {
        let data = vec![5.0, 10.0, 15.0, 20.0, 25.0];
        assert_eq!(median(&data), 15.0);
    }

    #[test]
    fn test_median_unsorted() {
        let data = vec![10.0, 30.0, 20.0, 50.0, 40.0];
        assert_eq!(median(&data), 30.0);
    }

    #[test]
    fn test_median_with_duplicates() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&data), 2.0);
    }

    use super::weighted_median;

    #[test]
    fn test_weighted_median_simple() {
        let data = vec![1.0, 2.0, 3.0];
        let weights = vec![0.2, 0.5, 0.3];
        assert_eq!(weighted_median(&data, &weights), 2.0);
    }

    #[test]
    fn test_weighted_median_even_weights() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.25, 0.25, 0.25, 0.25];
        assert_eq!(weighted_median(&data, &weights), 2.5);
    }

    #[test]
    fn test_weighted_median_single_element() {
        let data = vec![42.0];
        let weights = vec![1.0];
        assert_eq!(weighted_median(&data, &weights), 42.0);
    }

    #[test]
    #[should_panic(expected = "The length of data and weights must be the same")]
    fn test_weighted_median_mismatched_lengths() {
        let data = vec![1.0, 2.0, 3.0];
        let weights = vec![0.1, 0.2];
        weighted_median(&data, &weights);
    }

    #[test]
    fn test_weighted_median_all_same_elements() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(weighted_median(&data, &weights), 5.0);
    }

    #[test]
    #[should_panic(expected = "Weights must be non-negative")]
    fn test_weighted_median_negative_weights() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.2, -0.5, 0.5, 0.8];
        assert_eq!(weighted_median(&data, &weights), 4.0);
    }

    #[test]
    fn test_weighted_median_unsorted_data() {
        let data = vec![3.0, 1.0, 4.0, 2.0];
        let weights = vec![0.1, 0.3, 0.4, 0.2];
        assert_eq!(weighted_median(&data, &weights), 2.5);
    }

    #[test]
    fn test_weighted_median_with_zero_weights() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.0, 0.0, 1.0, 0.0];
        assert_eq!(weighted_median(&data, &weights), 3.0);
    }
}
