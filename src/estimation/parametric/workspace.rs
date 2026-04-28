use pharmsol::Equation;

use crate::algorithms::{Status, StopReason};
use crate::compile::OccasionDesign;
use crate::estimation::parametric::{
    IndividualEstimates, LikelihoodEstimates, ParametricIterationLog, ParametricPredictions,
    Population, ResidualErrorEstimates, UncertaintyEstimates,
};
use crate::output::shared::RunConfiguration;
use crate::results::FitResult;
use pharmsol::Data;

use super::state::{IndividualEffectsState, ParametricModelState};

#[derive(Debug)]
pub struct ParametricWorkspace<E: Equation> {
    state: ParametricModelState,
    individuals: IndividualEffectsState,
    equation: E,
    data: Data,
    population: Population,
    individual_estimates: IndividualEstimates,
    objf: f64,
    iterations: usize,
    status: Status,
    run_configuration: RunConfiguration,
    iteration_log: ParametricIterationLog,
    likelihoods: LikelihoodEstimates,
    uncertainty: UncertaintyEstimates,
    sigma: ResidualErrorEstimates,
    predictions: Option<ParametricPredictions>,
}

impl<E: Equation> ParametricWorkspace<E> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        state: ParametricModelState,
        individuals: IndividualEffectsState,
        equation: E,
        data: Data,
        population: Population,
        individual_estimates: IndividualEstimates,
        objf: f64,
        iterations: usize,
        status: Status,
        run_configuration: RunConfiguration,
        iteration_log: ParametricIterationLog,
        likelihoods: LikelihoodEstimates,
        uncertainty: UncertaintyEstimates,
        sigma: ResidualErrorEstimates,
        predictions: Option<ParametricPredictions>,
    ) -> Self {
        Self {
            state,
            individuals,
            equation,
            data,
            population,
            individual_estimates,
            objf,
            iterations,
            status,
            run_configuration,
            iteration_log,
            likelihoods,
            uncertainty,
            sigma,
            predictions,
        }
    }

    pub fn state(&self) -> &ParametricModelState {
        &self.state
    }

    pub fn population(&self) -> &Population {
        &self.population
    }

    pub fn mu(&self) -> &faer::Col<f64> {
        self.population.mu()
    }

    pub fn omega(&self) -> &faer::Mat<f64> {
        self.population.omega()
    }

    pub fn individual_estimates(&self) -> &IndividualEstimates {
        &self.individual_estimates
    }

    pub fn objf(&self) -> f64 {
        self.objf
    }

    pub fn best_objf(&self) -> f64 {
        self.likelihoods.best_objf().unwrap_or(self.objf)
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn converged(&self) -> bool {
        self.status == Status::Stop(StopReason::Converged)
    }

    pub fn status(&self) -> &Status {
        &self.status
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
        (
            self.run_configuration.runtime.idelta,
            self.run_configuration.runtime.tad,
        )
    }

    pub fn data(&self) -> &Data {
        &self.data
    }

    pub fn sigma(&self) -> &ResidualErrorEstimates {
        &self.sigma
    }

    pub fn predictions(&self) -> Option<&ParametricPredictions> {
        self.predictions.as_ref()
    }

    pub fn set_predictions(&mut self, predictions: ParametricPredictions) {
        self.predictions = Some(predictions);
    }

    pub fn individuals(&self) -> &IndividualEffectsState {
        &self.individuals
    }

    pub fn likelihoods(&self) -> &LikelihoodEstimates {
        &self.likelihoods
    }

    pub fn uncertainty(&self) -> &UncertaintyEstimates {
        &self.uncertainty
    }

    pub fn equation(&self) -> &E {
        &self.equation
    }

    pub fn standard_deviations(&self) -> faer::Col<f64> {
        self.population.standard_deviations()
    }

    pub fn cv_percent(&self) -> faer::Col<f64> {
        self.population.coefficient_of_variation()
    }

    pub fn correlation_matrix(&self) -> faer::Mat<f64> {
        self.population.correlation_matrix()
    }

    pub fn iteration_log(&self) -> &ParametricIterationLog {
        &self.iteration_log
    }

    pub fn write_covariates(&self) -> anyhow::Result<()> {
        use csv::WriterBuilder;
        use pharmsol::Event;

        tracing::debug!("Writing covariates...");

        let outputfile = crate::output::OutputFile::new(self.output_folder(), "covariates.csv")?;
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

        if covariate_names.is_empty() {
            return Ok(());
        }

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

                    let mut row: Vec<String> = vec![
                        subject.id().clone(),
                        time.to_string(),
                        occasion.index().to_string(),
                    ];

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

    pub fn write_population(&self) -> anyhow::Result<()> {
        use csv::WriterBuilder;

        let outputfile = crate::output::OutputFile::new(self.output_folder(), "population.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(outputfile.file());

        let header = ["parameter", "mu", "omega_diag", "sd", "cv_percent"];
        writer.write_record(header)?;

        let names = self.population.param_names();
        let sds = self.standard_deviations();
        let cvs = self.cv_percent();

        for (i, name) in names.iter().enumerate() {
            let row = vec![
                name.clone(),
                self.population.mu()[i].to_string(),
                self.population.omega()[(i, i)].to_string(),
                sds[i].to_string(),
                cvs[i].to_string(),
            ];
            writer.write_record(&row)?;
        }
        writer.flush()?;

        let outputfile = crate::output::OutputFile::new(self.output_folder(), "correlation.csv")?;
        let mut writer = WriterBuilder::new().from_writer(outputfile.file());
        let corr = self.correlation_matrix();
        let names = self.population.param_names();
        let mut header = vec!["".to_string()];
        header.extend(names.clone());
        writer.write_record(&header)?;
        for (i, name) in names.iter().enumerate() {
            let mut row = vec![name.clone()];
            for j in 0..corr.ncols() {
                row.push(format!("{:.4}", corr[(i, j)]));
            }
            writer.write_record(&row)?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn write_uncertainty(&self) -> anyhow::Result<()> {
        use csv::WriterBuilder;

        if !self.uncertainty.has_fim() && !self.uncertainty.has_standard_errors() {
            return Ok(());
        }

        let outputfile = crate::output::OutputFile::new(self.output_folder(), "uncertainty.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(outputfile.file());
        writer.write_record(["kind", "parameter", "value"])?;

        if let Some(method) = self.uncertainty.fim_method() {
            writer.write_record(["fim_method", "", &format!("{:?}", method)])?;
        }

        for (index, name) in self.population.param_names().iter().enumerate() {
            writer.write_record(["mu", name, &format!("{:.6}", self.population.mu()[index])])?;

            if let Some(se_mu) = self.uncertainty.se_mu() {
                writer.write_record(["se_mu", name, &format!("{:.6}", se_mu[index])])?;
            }

            if let Some(rse_mu) = self.uncertainty.rse_mu() {
                writer.write_record(["rse_mu", name, &format!("{:.6}", rse_mu[index])])?;
            }

            if let Some(se_omega) = self.uncertainty.se_omega() {
                writer.write_record([
                    "se_omega_diag",
                    name,
                    &format!("{:.6}", se_omega[(index, index)]),
                ])?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    pub fn write_individual_parameters(&self) -> anyhow::Result<()> {
        use csv::WriterBuilder;

        let outputfile =
            crate::output::OutputFile::new(self.output_folder(), "individual_parameters.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(outputfile.file());

        let names = self.population.param_names();
        let mut header = vec!["id".to_string()];
        for name in &names {
            header.push(format!("psi_{}", name));
        }
        writer.write_record(&header)?;

        for ind in self.individual_estimates.iter() {
            let mut row = vec![ind.subject_id().to_string()];
            for i in 0..ind.npar() {
                row.push(ind.psi()[i].to_string());
            }
            writer.write_record(&row)?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn write_individual_effects(&self) -> anyhow::Result<()> {
        use csv::WriterBuilder;

        let outputfile =
            crate::output::OutputFile::new(self.output_folder(), "individual_effects.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(outputfile.file());

        let names = self.population.param_names();
        let mut header = vec!["id".to_string()];
        for name in &names {
            header.push(format!("eta_{}", name));
        }
        if self
            .individual_estimates
            .iter()
            .any(|i| i.objective_function().is_some())
        {
            header.push("objf".to_string());
        }
        writer.write_record(&header)?;

        for ind in self.individual_estimates.iter() {
            let mut row = vec![ind.subject_id().to_string()];
            for i in 0..ind.npar() {
                row.push(ind.eta()[i].to_string());
            }
            if let Some(objf) = ind.objective_function() {
                row.push(objf.to_string());
            }
            writer.write_record(&row)?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn write_iteration_log(&self) -> anyhow::Result<()> {
        self.iteration_log.write(
            self.output_folder(),
            &self.run_configuration.parameter_names,
        )
    }

    pub fn with_compiled_state(
        mut self,
        compiled_state: ParametricModelState,
        occasions: &[OccasionDesign],
    ) -> Self {
        self.individuals = self.individuals.with_occasion_design(
            occasions,
            &compiled_state.variability,
            self.population.npar(),
        );
        self.state = compiled_state.merged(self.state);
        self
    }

    pub fn into_fit_result(self) -> FitResult<E> {
        FitResult::Parametric(self)
    }
}
