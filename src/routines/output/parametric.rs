//! Parametric algorithm result structures
//!
//! This module defines the result/output structures for parametric population
//! pharmacokinetic algorithms (SAEM, FOCE, etc.).
//!
//! # Key Structures
//!
//! - [`ParametricResult`]: Main result container for parametric algorithms
//! - [`ParametricCycleLog`]: Iteration history and convergence tracking
//! - [`LikelihoodEstimates`]: Log-likelihood computed by different methods
//!
//! # Output Files
//!
//! When `write_outputs()` is called, the following files are created:
//!
//! | File | Description |
//! |------|-------------|
//! | `population.csv` | Population parameters (μ, Ω diag, SD, CV%) |
//! | `correlation.csv` | Correlation matrix from Ω |
//! | `individual.csv` | Individual estimates (η, ψ) per subject |
//! | `iterations.csv` | Parameter history per iteration |
//! | `pred.csv` | Predictions (PPRED, IPRED) and residuals |
//! | `statistics.csv` | Summary stats (LL, AIC, BIC, shrinkage) |
//! | `sigma.csv` | Residual error estimates |
//! | `covs.csv` | Subject covariates |
//! | `settings.json` | Run configuration |

use anyhow::{Context, Result};
use csv::WriterBuilder;
use faer::{Col, Mat};
use pharmsol::{Data, Equation, Event};
use serde::{Deserialize, Serialize};

use crate::algorithms::{Status, StopReason};
use crate::routines::settings::Settings;
use crate::structs::parametric::{IndividualEstimates, Population};

use super::parametric_predictions::ParametricPredictions;
use super::parametric_statistics::{ParametricStatistics, ResidualErrorEstimates};
use super::OutputFile;

/// Likelihood estimates computed by different methods
///
/// SAEM provides an approximate likelihood during optimization, but more accurate
/// estimates can be computed post-hoc using various methods.
///
/// # R saemix Correspondence
///
/// | Method | R saemix function | Description |
/// |--------|-------------------|-------------|
/// | Linearization | `compute.LLlin()` | First-order Taylor approximation |
/// | Importance Sampling | `compute.LLis()` | Monte Carlo integration |
/// | Gaussian Quadrature | `compute.LLgq()` | Numerical integration |
#[derive(Debug, Clone, Default, Serialize)]
pub struct LikelihoodEstimates {
    /// Log-likelihood by linearization (first-order approximation)
    /// Fast but may be biased for highly nonlinear models
    pub ll_linearization: Option<f64>,

    /// Log-likelihood by importance sampling
    /// More accurate than linearization, requires MCMC samples
    pub ll_importance_sampling: Option<f64>,

    /// Log-likelihood by Gaussian quadrature
    /// Most accurate for low-dimensional problems
    pub ll_gaussian_quadrature: Option<f64>,

    /// Number of samples used for importance sampling
    pub is_n_samples: Option<usize>,

    /// Number of quadrature points used
    pub gq_n_points: Option<usize>,
}

impl LikelihoodEstimates {
    /// Create new empty likelihood estimates
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the best available likelihood estimate
    ///
    /// Prefers Gaussian quadrature > Importance sampling > Linearization
    pub fn best_estimate(&self) -> Option<f64> {
        self.ll_gaussian_quadrature
            .or(self.ll_importance_sampling)
            .or(self.ll_linearization)
    }

    /// Get -2LL (objective function) from best available method
    pub fn best_objf(&self) -> Option<f64> {
        self.best_estimate().map(|ll| -2.0 * ll)
    }
}

/// Standard errors and uncertainty quantification
///
/// Contains the Fisher Information Matrix (FIM) and derived standard errors
/// for population parameters.
#[derive(Debug, Clone, Default)]
pub struct UncertaintyEstimates {
    /// Fisher Information Matrix
    /// Dimensions: (n_fixed + n_omega_elements) × (n_fixed + n_omega_elements)
    pub fim: Option<Mat<f64>>,

    /// Inverse of FIM (variance-covariance of estimates)
    pub fim_inverse: Option<Mat<f64>>,

    /// Standard errors of μ (population mean parameters)
    pub se_mu: Option<Col<f64>>,

    /// Standard errors of Ω elements (variance components)
    pub se_omega: Option<Mat<f64>>,

    /// Relative standard errors of μ (SE/estimate × 100%)
    pub rse_mu: Option<Col<f64>>,

    /// Method used to compute FIM
    pub fim_method: Option<FimMethod>,
}

/// Method used to compute Fisher Information Matrix
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FimMethod {
    /// Observed FIM from Hessian of log-likelihood
    Observed,
    /// Expected FIM (Louis' method for EM algorithms)
    Expected,
    /// Stochastic approximation of FIM
    StochasticApproximation,
    /// Linearization-based FIM
    Linearization,
}

impl UncertaintyEstimates {
    /// Create new empty uncertainty estimates
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if FIM has been computed
    pub fn has_fim(&self) -> bool {
        self.fim.is_some()
    }

    /// Check if standard errors are available
    pub fn has_standard_errors(&self) -> bool {
        self.se_mu.is_some()
    }
}

/// Result from a parametric algorithm run
///
/// Contains all the information needed to summarize the fit, make predictions,
/// and perform post-hoc analyses.
#[derive(Debug)]
pub struct ParametricResult<E: Equation> {
    /// The equation/model used
    equation: E,
    /// The input data
    data: Data,
    /// Final population parameter estimates (μ, Ω)
    population: Population,
    /// Individual parameter estimates (EBEs)
    individual_estimates: IndividualEstimates,
    /// Final objective function value (-2LL) from algorithm
    objf: f64,
    /// Number of iterations
    iterations: usize,
    /// Final algorithm status
    status: Status,
    /// Algorithm settings
    settings: Settings,
    /// Iteration history
    iteration_log: ParametricIterationLog,
    /// Likelihood estimates by different methods (computed post-hoc)
    likelihood_estimates: LikelihoodEstimates,
    /// Uncertainty quantification (FIM, standard errors)
    uncertainty: UncertaintyEstimates,
    /// Residual error estimates (σ)
    sigma: ResidualErrorEstimates,
    /// Cached predictions (computed on demand)
    predictions: Option<ParametricPredictions>,
}

impl<E: Equation> ParametricResult<E> {
    /// Create a new parametric result
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        equation: E,
        data: Data,
        population: Population,
        individual_estimates: IndividualEstimates,
        objf: f64,
        iterations: usize,
        status: Status,
        settings: Settings,
        iteration_log: ParametricIterationLog,
    ) -> Self {
        Self {
            equation,
            data,
            population,
            individual_estimates,
            objf,
            iterations,
            status,
            settings,
            iteration_log,
            likelihood_estimates: LikelihoodEstimates::new(),
            uncertainty: UncertaintyEstimates::new(),
            sigma: ResidualErrorEstimates::default(),
            predictions: None,
        }
    }

    /// Create a new parametric result with sigma estimates
    #[allow(clippy::too_many_arguments)]
    pub fn with_sigma(
        equation: E,
        data: Data,
        population: Population,
        individual_estimates: IndividualEstimates,
        objf: f64,
        iterations: usize,
        status: Status,
        settings: Settings,
        iteration_log: ParametricIterationLog,
        sigma: ResidualErrorEstimates,
    ) -> Self {
        Self {
            equation,
            data,
            population,
            individual_estimates,
            objf,
            iterations,
            status,
            settings,
            iteration_log,
            likelihood_estimates: LikelihoodEstimates::new(),
            uncertainty: UncertaintyEstimates::new(),
            sigma,
            predictions: None,
        }
    }

    // ========== Getters ==========

    /// Get the final population parameters
    pub fn population(&self) -> &Population {
        &self.population
    }

    /// Get the population mean (μ)
    pub fn mu(&self) -> &faer::Col<f64> {
        self.population.mu()
    }

    /// Get the population covariance (Ω)
    pub fn omega(&self) -> &faer::Mat<f64> {
        self.population.omega()
    }

    /// Get the individual estimates
    pub fn individual_estimates(&self) -> &IndividualEstimates {
        &self.individual_estimates
    }

    /// Get the final objective function value from the algorithm
    pub fn objf(&self) -> f64 {
        self.objf
    }

    /// Get the best available objective function (-2LL)
    ///
    /// Prefers post-hoc likelihood estimates over the algorithm's value
    pub fn best_objf(&self) -> f64 {
        self.likelihood_estimates.best_objf().unwrap_or(self.objf)
    }

    /// Get the number of iterations
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Check if the algorithm converged
    pub fn converged(&self) -> bool {
        self.status == Status::Stop(StopReason::Converged)
    }

    /// Get the algorithm status
    pub fn status(&self) -> &Status {
        &self.status
    }

    /// Get the settings
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Get the data
    pub fn data(&self) -> &Data {
        &self.data
    }

    /// Get the iteration log
    pub fn iteration_log(&self) -> &ParametricIterationLog {
        &self.iteration_log
    }

    /// Get the likelihood estimates
    pub fn likelihood_estimates(&self) -> &LikelihoodEstimates {
        &self.likelihood_estimates
    }

    /// Get the uncertainty estimates
    pub fn uncertainty(&self) -> &UncertaintyEstimates {
        &self.uncertainty
    }

    /// Get standard errors of μ if available
    pub fn se_mu(&self) -> Option<&Col<f64>> {
        self.uncertainty.se_mu.as_ref()
    }

    /// Get the Fisher Information Matrix if available
    pub fn fim(&self) -> Option<&Mat<f64>> {
        self.uncertainty.fim.as_ref()
    }

    /// Get the equation/model
    pub fn equation(&self) -> &E {
        &self.equation
    }

    /// Get the residual error estimates
    pub fn sigma(&self) -> &ResidualErrorEstimates {
        &self.sigma
    }

    /// Get the cached predictions if available
    pub fn predictions(&self) -> Option<&ParametricPredictions> {
        self.predictions.as_ref()
    }

    // ========== Setters for post-hoc computation ==========

    /// Set likelihood estimates (computed post-hoc)
    pub fn set_likelihood_estimates(&mut self, estimates: LikelihoodEstimates) {
        self.likelihood_estimates = estimates;
    }

    /// Set uncertainty estimates (FIM, standard errors)
    pub fn set_uncertainty(&mut self, uncertainty: UncertaintyEstimates) {
        self.uncertainty = uncertainty;
    }

    /// Set residual error estimates
    pub fn set_sigma(&mut self, sigma: ResidualErrorEstimates) {
        self.sigma = sigma;
    }

    // ========== Derived Quantities ==========

    /// Get the standard errors of population parameters (from Ω diagonal)
    pub fn standard_deviations(&self) -> faer::Col<f64> {
        self.population.standard_deviations()
    }

    /// Get the coefficient of variation for each parameter
    pub fn cv_percent(&self) -> faer::Col<f64> {
        self.population.coefficient_of_variation()
    }

    /// Get the correlation matrix derived from Ω
    pub fn correlation_matrix(&self) -> faer::Mat<f64> {
        self.population.correlation_matrix()
    }

    /// Calculate shrinkage for each parameter
    pub fn shrinkage(&self) -> Option<faer::Col<f64>> {
        let n = self.population.npar();
        let pop_var = faer::Col::from_fn(n, |i| self.population.omega()[(i, i)]);
        self.individual_estimates.shrinkage(&pop_var)
    }

    /// Get AIC (Akaike Information Criterion)
    ///
    /// AIC = -2LL + 2k where k is the number of estimated parameters
    /// Uses the best available likelihood estimate
    pub fn aic(&self) -> f64 {
        let n_params = self.population.npar();
        // Population mean parameters + covariance parameters
        let n_fixed = n_params;
        let n_random = n_params * (n_params + 1) / 2; // Lower triangle of Ω
        let k = n_fixed + n_random;
        self.best_objf() + 2.0 * k as f64
    }

    /// Get BIC (Bayesian Information Criterion)
    ///
    /// BIC = -2LL + k * ln(n) where k is number of parameters, n is number of subjects
    /// Uses the best available likelihood estimate
    pub fn bic(&self) -> f64 {
        let n_subjects = self.data.subjects().len();
        let n_params = self.population.npar();
        let n_fixed = n_params;
        let n_random = n_params * (n_params + 1) / 2;
        let k = n_fixed + n_random;
        self.best_objf() + (k as f64) * (n_subjects as f64).ln()
    }

    // ========== Output Methods ==========

    /// Write all outputs to files
    ///
    /// Creates the following output files:
    /// - `population.csv` - Population parameters (μ, Ω)
    /// - `correlation.csv` - Correlation matrix from Ω
    /// - `individual.csv` - Individual parameter estimates
    /// - `iterations.csv` - Parameter history per iteration
    /// - `pred.csv` - Predictions and residuals (if predictions calculated)
    /// - `statistics.csv` - Summary statistics (LL, AIC, BIC, shrinkage)
    /// - `sigma.csv` - Residual error estimates
    /// - `covs.csv` - Subject covariates
    /// - `settings.json` - Run configuration
    pub fn write_outputs(&mut self) -> Result<()> {
        if !self.settings.output().write {
            return Ok(());
        }

        tracing::debug!("Writing outputs to {:?}", self.settings.output().path);

        // Core output files (always written)
        self.write_population()
            .context("Failed to write population")?;
        self.write_individual_estimates()
            .context("Failed to write individual estimates")?;
        self.write_iteration_log()
            .context("Failed to write iteration log")?;

        // Predictions (calculate if not already done)
        let idelta = self.settings.predictions().idelta;
        let tad = self.settings.predictions().tad;
        self.calculate_predictions(idelta, tad)
            .context("Failed to calculate predictions")?;
        if let Some(ref preds) = self.predictions {
            preds.write(&self.settings)
                .context("Failed to write predictions")?;
        }

        // Statistics summary
        self.write_statistics()
            .context("Failed to write statistics")?;

        // Residual error estimates
        self.sigma.write(&self.settings)
            .context("Failed to write sigma")?;

        // Covariates
        self.write_covariates()
            .context("Failed to write covariates")?;

        // Settings
        self.settings.write()?;

        Ok(())
    }

    /// Calculate and cache predictions
    ///
    /// # Arguments
    /// * `idelta` - Time increment for prediction grid expansion
    /// * `tad` - Time after dose for grid expansion
    pub fn calculate_predictions(&mut self, idelta: f64, tad: f64) -> Result<()> {
        // Get sigma for IWRES calculation (use additive if available)
        let sigma_val = self.sigma.additive.or(self.sigma.proportional);

        let predictions = ParametricPredictions::calculate(
            &self.equation,
            &self.data,
            &self.population,
            &self.individual_estimates,
            sigma_val,
            idelta,
            tad,
        )?;

        self.predictions = Some(predictions);
        Ok(())
    }

    /// Write statistics summary file
    pub fn write_statistics(&self) -> Result<()> {
        // Count observations
        let n_obs: usize = self
            .data
            .subjects()
            .iter()
            .flat_map(|s| s.occasions())
            .flat_map(|o| o.events())
            .filter(|e| matches!(e, Event::Observation(_)))
            .count();

        let stats = ParametricStatistics::from_result(
            &self.population,
            &self.individual_estimates,
            self.objf,
            self.iterations,
            self.converged(),
            self.data.len(),
            n_obs,
            self.likelihood_estimates.ll_importance_sampling,
            self.likelihood_estimates.ll_linearization,
            self.likelihood_estimates.ll_gaussian_quadrature,
            self.sigma.as_vec(),
        );

        stats.write(&self.settings)?;
        stats.write_shrinkage(&self.settings, &self.population.param_names())?;

        Ok(())
    }

    /// Write covariates to a CSV file
    pub fn write_covariates(&self) -> Result<()> {
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
        covariate_names.sort();

        // If no covariates, skip writing
        if covariate_names.is_empty() {
            tracing::debug!("No covariates found, skipping covs.csv");
            return Ok(());
        }

        // Write header
        let mut headers = vec!["id", "time", "block"];
        headers.extend(covariate_names.iter().map(|s| s.as_str()));
        writer.write_record(&headers)?;

        // Write data rows
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
        tracing::debug!("Covariates written to {:?}", outputfile.relative_path());

        Ok(())
    }

    /// Write the population parameters to CSV
    pub fn write_population(&self) -> Result<()> {
        tracing::debug!("Writing population parameters...");

        let outputfile = OutputFile::new(&self.settings.output().path, "population.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(&outputfile.file);

        // Write header
        let mut header = vec!["parameter".to_string()];
        header.push("mu".to_string());
        header.push("omega_diag".to_string());
        header.push("sd".to_string());
        header.push("cv_percent".to_string());
        writer.write_record(&header)?;

        // Write each parameter
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
        tracing::debug!(
            "Population parameters written to {:?}",
            outputfile.relative_path()
        );

        // Write correlation matrix
        self.write_correlation_matrix()?;

        Ok(())
    }

    /// Write the correlation matrix
    fn write_correlation_matrix(&self) -> Result<()> {
        let outputfile = OutputFile::new(&self.settings.output().path, "correlation.csv")?;
        let mut writer = WriterBuilder::new().from_writer(&outputfile.file);

        let corr = self.correlation_matrix();
        let names = self.population.param_names();

        // Header
        let mut header = vec!["".to_string()];
        header.extend(names.clone());
        writer.write_record(&header)?;

        // Rows
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

    /// Write individual parameter estimates
    pub fn write_individual_estimates(&self) -> Result<()> {
        tracing::debug!("Writing individual parameter estimates...");

        let outputfile = OutputFile::new(&self.settings.output().path, "individual.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(&outputfile.file);

        // Header: id, eta_1, eta_2, ..., psi_1, psi_2, ...
        let names = self.population.param_names();
        let mut header = vec!["id".to_string()];
        for name in &names {
            header.push(format!("eta_{}", name));
        }
        for name in &names {
            header.push(format!("psi_{}", name));
        }
        if self.individual_estimates.iter().any(|i| i.objective_function().is_some()) {
            header.push("objf".to_string());
        }
        writer.write_record(&header)?;

        // Write each individual
        for ind in self.individual_estimates.iter() {
            let mut row = vec![ind.subject_id().to_string()];

            // Eta values
            for i in 0..ind.npar() {
                row.push(ind.eta()[i].to_string());
            }

            // Psi values
            for i in 0..ind.npar() {
                row.push(ind.psi()[i].to_string());
            }

            // Objective function if available
            if let Some(objf) = ind.objective_function() {
                row.push(objf.to_string());
            }

            writer.write_record(&row)?;
        }

        writer.flush()?;
        tracing::debug!(
            "Individual estimates written to {:?}",
            outputfile.relative_path()
        );

        Ok(())
    }

    /// Write the iteration log
    pub fn write_iteration_log(&self) -> Result<()> {
        self.iteration_log.write(&self.settings)
    }
}

/// Iteration log for parametric algorithms
#[derive(Debug, Clone, Default)]
pub struct ParametricIterationLog {
    /// Iteration numbers
    iterations: Vec<usize>,
    /// Objective function values
    objf: Vec<f64>,
    /// Population mean at each iteration (flattened)
    mu_history: Vec<Vec<f64>>,
    /// Population omega diagonal at each iteration
    omega_diag_history: Vec<Vec<f64>>,
    /// Status at each iteration
    status: Vec<String>,
}

impl ParametricIterationLog {
    /// Create a new empty log
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an iteration to the log
    pub fn log_iteration(
        &mut self,
        iteration: usize,
        objf: f64,
        population: &Population,
        status: &Status,
    ) {
        self.iterations.push(iteration);
        self.objf.push(objf);

        // Store mu
        let mu: Vec<f64> = (0..population.npar())
            .map(|i| population.mu()[i])
            .collect();
        self.mu_history.push(mu);

        // Store omega diagonal
        let omega_diag: Vec<f64> = (0..population.npar())
            .map(|i| population.omega()[(i, i)])
            .collect();
        self.omega_diag_history.push(omega_diag);

        self.status.push(format!("{:?}", status));
    }

    /// Get the number of logged iterations
    pub fn len(&self) -> usize {
        self.iterations.len()
    }

    /// Check if the log is empty
    pub fn is_empty(&self) -> bool {
        self.iterations.is_empty()
    }

    /// Get the objective function history
    pub fn objf_history(&self) -> &[f64] {
        &self.objf
    }

    /// Write the log to a CSV file
    pub fn write(&self, settings: &Settings) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }

        let outputfile = OutputFile::new(&settings.output().path, "iterations.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(&outputfile.file);

        // Determine parameter names
        let n_params = self.mu_history.first().map(|v| v.len()).unwrap_or(0);
        let param_names = settings.parameters().names();

        // Header
        let mut header = vec!["iteration".to_string(), "objf".to_string()];
        for name in &param_names {
            header.push(format!("mu_{}", name));
        }
        for name in &param_names {
            header.push(format!("omega_{}", name));
        }
        header.push("status".to_string());
        writer.write_record(&header)?;

        // Write each iteration
        for i in 0..self.iterations.len() {
            let mut row = vec![
                self.iterations[i].to_string(),
                format!("{:.6}", self.objf[i]),
            ];

            for j in 0..n_params {
                row.push(format!("{:.6}", self.mu_history[i].get(j).unwrap_or(&0.0)));
            }

            for j in 0..n_params {
                row.push(format!("{:.6}", self.omega_diag_history[i].get(j).unwrap_or(&0.0)));
            }

            row.push(self.status[i].clone());
            writer.write_record(&row)?;
        }

        writer.flush()?;
        tracing::debug!(
            "Iteration log written to {:?}",
            outputfile.relative_path()
        );

        Ok(())
    }
}

impl Serialize for ParametricIterationLog {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("ParametricIterationLog", 4)?;
        state.serialize_field("iterations", &self.iterations)?;
        state.serialize_field("objf", &self.objf)?;
        state.serialize_field("status", &self.status)?;
        state.end()
    }
}
