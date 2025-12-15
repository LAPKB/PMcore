//! Parametric algorithm result structures
//!
//! This module defines the result/output structures for parametric population
//! pharmacokinetic algorithms (SAEM, FOCE, etc.).
//!
//! # Key Structures
//!
//! - [`ParametricResult`]: Main result container for parametric algorithms
//! - [`ParametricCycleLog`]: Iteration history and convergence tracking

use anyhow::{Context, Result};
use csv::WriterBuilder;
use pharmsol::{Data, Equation};
use serde::Serialize;

use crate::algorithms::{Status, StopReason};
use crate::routines::settings::Settings;
use crate::structs::parametric::{IndividualEstimates, Population};

use super::OutputFile;

/// Result from a parametric algorithm run
///
/// Contains all the information needed to summarize the fit, make predictions,
/// and perform post-hoc analyses.
#[derive(Debug)]
pub struct ParametricResult<E: Equation> {
    /// The equation/model used
    #[allow(dead_code)]
    equation: E,
    /// The input data
    data: Data,
    /// Final population parameter estimates (μ, Ω)
    population: Population,
    /// Individual parameter estimates (EBEs)
    individual_estimates: IndividualEstimates,
    /// Final objective function value (-2LL)
    objf: f64,
    /// Number of iterations
    iterations: usize,
    /// Final algorithm status
    status: Status,
    /// Algorithm settings
    settings: Settings,
    /// Iteration history
    iteration_log: ParametricIterationLog,
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

    /// Get the final objective function value
    pub fn objf(&self) -> f64 {
        self.objf
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
    pub fn aic(&self) -> f64 {
        let n_params = self.population.npar();
        // Population mean parameters + covariance parameters
        let n_fixed = n_params;
        let n_random = n_params * (n_params + 1) / 2; // Lower triangle of Ω
        let k = n_fixed + n_random;
        self.objf + 2.0 * k as f64
    }

    /// Get BIC (Bayesian Information Criterion)
    ///
    /// BIC = -2LL + k * ln(n) where k is number of parameters, n is number of subjects
    pub fn bic(&self) -> f64 {
        let n_subjects = self.data.subjects().len();
        let n_params = self.population.npar();
        let n_fixed = n_params;
        let n_random = n_params * (n_params + 1) / 2;
        let k = n_fixed + n_random;
        self.objf + (k as f64) * (n_subjects as f64).ln()
    }

    // ========== Output Methods ==========

    /// Write all outputs to files
    pub fn write_outputs(&self) -> Result<()> {
        if !self.settings.output().write {
            return Ok(());
        }

        tracing::debug!("Writing outputs to {:?}", self.settings.output().path);

        self.write_population().context("Failed to write population")?;
        self.write_individual_estimates()
            .context("Failed to write individual estimates")?;
        self.write_iteration_log()
            .context("Failed to write iteration log")?;
        self.settings.write()?;

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
