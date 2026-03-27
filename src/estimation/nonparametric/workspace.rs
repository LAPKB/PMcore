use pharmsol::Equation;

use crate::algorithms::{Status, StopReason};
use crate::estimation::nonparametric::{posterior, CycleLog, NPPredictions, Posterior, Psi, Theta, Weights};
use crate::output::shared::RunConfiguration;
use crate::results::FitResult;
use pharmsol::Data;

#[derive(Debug)]
pub struct NonparametricWorkspace<E: Equation> {
    equation: E,
    data: Data,
    theta: Theta,
    psi: Psi,
    weights: Weights,
    objf: f64,
    cycles: usize,
    status: Status,
    run_configuration: RunConfiguration,
    cyclelog: CycleLog,
    predictions: Option<NPPredictions>,
    posterior: Posterior,
}

impl<E: Equation> NonparametricWorkspace<E> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        equation: E,
        data: Data,
        theta: Theta,
        psi: Psi,
        weights: Weights,
        objf: f64,
        cycles: usize,
        status: Status,
        run_configuration: RunConfiguration,
        cyclelog: CycleLog,
    ) -> anyhow::Result<Self> {
        let posterior = posterior::posterior(&psi, &weights)?;

        Ok(Self {
            equation,
            data,
            theta,
            psi,
            weights,
            objf,
            cycles,
            status,
            run_configuration,
            cyclelog,
            predictions: None,
            posterior,
        })
    }

    pub fn cycles(&self) -> usize {
        self.cycles
    }

    pub fn objf(&self) -> f64 {
        self.objf
    }

    pub fn converged(&self) -> bool {
        self.status == Status::Stop(StopReason::Converged)
    }

    pub fn get_theta(&self) -> &Theta {
        &self.theta
    }

    pub fn data(&self) -> &Data {
        &self.data
    }

    pub fn cycle_log(&self) -> &CycleLog {
        &self.cyclelog
    }

    pub(crate) fn run_configuration(&self) -> &RunConfiguration {
        &self.run_configuration
    }

    pub(crate) fn algorithm(&self) -> crate::algorithms::Algorithm {
        self.run_configuration.algorithm
    }

    pub(crate) fn output_folder(&self) -> &str {
        self.run_configuration.output_path()
    }

    pub(crate) fn should_write_outputs(&self) -> bool {
        self.run_configuration.should_write_outputs()
    }

    pub(crate) fn prediction_interval(&self) -> (f64, f64) {
        (self.run_configuration.runtime.idelta, self.run_configuration.runtime.tad)
    }

    pub fn predictions(&self) -> Option<&NPPredictions> {
        self.predictions.as_ref()
    }

    pub fn psi(&self) -> &Psi {
        &self.psi
    }

    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    pub fn posterior(&self) -> &Posterior {
        &self.posterior
    }

    pub fn calculate_predictions(&mut self, idelta: f64, tad: f64) -> anyhow::Result<()> {
        let predictions = NPPredictions::calculate(
            &self.equation,
            &self.data,
            &self.theta,
            &self.weights,
            &self.posterior,
            idelta,
            tad,
        )?;
        self.predictions = Some(predictions);
        Ok(())
    }

    pub fn write_theta(&self) -> anyhow::Result<()> {
        use anyhow::{bail, Context};
        use csv::WriterBuilder;

        tracing::debug!("Writing population parameter distribution...");

        let w: Vec<f64> = self.weights.to_vec();
        if w.len() != self.theta.matrix().nrows() {
            bail!(
                "Number of weights ({}) and number of support points ({}) do not match.",
                w.len(),
                self.theta.matrix().nrows()
            );
        }

        let outputfile = crate::output::OutputFile::new(
            self.output_folder(),
            "theta.csv",
        )
        .context("Failed to create output file for theta")?;

        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(outputfile.file());

        let mut theta_header = self.run_configuration.parameter_names.clone();
        theta_header.push("prob".to_string());
        writer.write_record(&theta_header)?;

        for (theta_row, &w_val) in self.theta.matrix().row_iter().zip(w.iter()) {
            let mut row: Vec<String> = theta_row.iter().map(|&val| val.to_string()).collect();
            row.push(w_val.to_string());
            writer.write_record(&row)?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn write_posterior(&self) -> anyhow::Result<()> {
        use csv::WriterBuilder;

        tracing::debug!("Writing posterior parameter probabilities...");

        let outputfile = crate::output::OutputFile::new(
            self.output_folder(),
            "posterior.csv",
        )?;

        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(outputfile.file());

        writer.write_field("id")?;
        writer.write_field("point")?;
        self.theta.param_names().iter().for_each(|name| {
            writer.write_field(name).unwrap();
        });
        writer.write_field("prob")?;
        writer.write_record(None::<&[u8]>)?;

        let subjects = self.data.subjects();
        self.posterior
            .matrix()
            .row_iter()
            .enumerate()
            .for_each(|(i, row)| {
                let subject = subjects.get(i).unwrap();
                let id = subject.id();

                row.iter().enumerate().for_each(|(spp, prob)| {
                    writer.write_field(id.clone()).unwrap();
                    writer.write_field(spp.to_string()).unwrap();

                    self.theta.matrix().row(spp).iter().for_each(|val| {
                        writer.write_field(val.to_string()).unwrap();
                    });

                    writer.write_field(prob.to_string()).unwrap();
                    writer.write_record(None::<&[u8]>).unwrap();
                });
            });

        writer.flush()?;
        Ok(())
    }

    pub fn write_predictions(&mut self, idelta: f64, tad: f64) -> anyhow::Result<()> {
        use csv::WriterBuilder;

        tracing::debug!("Writing predictions...");
        self.calculate_predictions(idelta, tad)?;

        let predictions = self
            .predictions
            .as_ref()
            .expect("Predictions should have been calculated, but are of type None.");

        let outputfile = crate::output::OutputFile::new(self.output_folder(), "pred.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(outputfile.file());

        for row in predictions.predictions() {
            writer.serialize(row)?;
        }

        writer.flush()?;
        Ok(())
    }

    pub fn write_covs(&self) -> anyhow::Result<()> {
        use csv::WriterBuilder;
        use pharmsol::Event;

        tracing::debug!("Writing covariates...");
        let outputfile = crate::output::OutputFile::new(self.output_folder(), "covs.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(outputfile.file());

        let mut covariate_names = std::collections::HashSet::new();
        for subject in self.data.subjects() {
            for occasion in subject.occasions() {
                let covmap = occasion.covariates().covariates();
                for cov_name in covmap.keys() {
                    covariate_names.insert(cov_name.clone());
                }
            }
        }
        let mut covariate_names: Vec<String> = covariate_names.into_iter().collect();
        covariate_names.sort();

        let mut headers = vec!["id", "time", "block"];
        headers.extend(covariate_names.iter().map(|s| s.as_str()));
        writer.write_record(&headers)?;

        for subject in self.data.subjects() {
            for occasion in subject.occasions() {
                let covmap = occasion.covariates().covariates();

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
        Ok(())
    }

    pub fn into_fit_result(self) -> FitResult<E> {
        FitResult::Nonparametric(self)
    }
}