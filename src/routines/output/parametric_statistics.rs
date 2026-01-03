//! Statistical summaries for parametric algorithm results
//!
//! This module provides functions to compute and write statistical summaries
//! including likelihood estimates, information criteria, shrinkage, and other
//! diagnostic measures.
//!
//! # R saemix Correspondence
//!
//! | PMcore | R saemix | Description |
//! |--------|----------|-------------|
//! | `ll_is` | `ll.is` | Log-likelihood by importance sampling |
//! | `ll_lin` | `ll.lin` | Log-likelihood by linearization |
//! | `aic` | `aic.is` | Akaike Information Criterion |
//! | `bic` | `bic.is` | Bayesian Information Criterion |
//! | `shrinkage` | `cond.shrinkage` | Eta shrinkage |

use anyhow::Result;
use csv::WriterBuilder;
use serde::Serialize;

use crate::routines::output::OutputFile;
use crate::routines::settings::Settings;
use crate::structs::parametric::{IndividualEstimates, Population};

/// Statistical summary for parametric results
///
/// Contains all the key statistical measures that should be reported
/// after a parametric algorithm run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ParametricStatistics {
    // ========== Dataset Info ==========
    /// Number of subjects
    pub n_subjects: usize,
    /// Number of observations
    pub n_observations: usize,
    /// Number of estimated fixed effect parameters
    pub n_fixed: usize,
    /// Number of random effect variance parameters
    pub n_random: usize,
    /// Total number of estimated parameters
    pub n_total_params: usize,

    // ========== Convergence ==========
    /// Number of iterations completed
    pub iterations: usize,
    /// Whether algorithm converged
    pub converged: bool,

    // ========== Likelihood ==========
    /// Objective function value (-2LL) from algorithm
    pub objf: f64,
    /// Log-likelihood by importance sampling
    pub ll_is: Option<f64>,
    /// Log-likelihood by linearization
    pub ll_lin: Option<f64>,
    /// Log-likelihood by Gaussian quadrature
    pub ll_gq: Option<f64>,

    // ========== Information Criteria ==========
    /// AIC using best available LL
    pub aic: f64,
    /// BIC using best available LL
    pub bic: f64,

    // ========== Shrinkage ==========
    /// Eta shrinkage for each parameter
    pub eta_shrinkage: Vec<f64>,
    /// Overall eta shrinkage (average)
    pub eta_shrinkage_overall: f64,

    // ========== Residual Error ==========
    /// Estimated residual error parameter(s)
    /// For additive: σ (SD)
    /// For proportional: σ_prop
    /// For combined: (σ_add, σ_prop)
    pub sigma: Vec<f64>,

    // ========== Parameter Estimates ==========
    /// Population means (μ)
    pub mu: Vec<f64>,
    /// Population variances (diag(Ω))
    pub omega_diag: Vec<f64>,
    /// Population standard deviations (sqrt(diag(Ω)))
    pub omega_sd: Vec<f64>,
    /// Coefficient of variation (%) for each parameter
    pub cv_percent: Vec<f64>,
}

impl ParametricStatistics {
    /// Create new statistics from result components
    pub fn from_result(
        population: &Population,
        individual_estimates: &IndividualEstimates,
        objf: f64,
        iterations: usize,
        converged: bool,
        n_subjects: usize,
        n_observations: usize,
        ll_is: Option<f64>,
        ll_lin: Option<f64>,
        ll_gq: Option<f64>,
        sigma: Vec<f64>,
    ) -> Self {
        let n_fixed = population.npar();
        let n_random = n_fixed; // Assuming diagonal Omega
        let n_total = n_fixed + n_random + sigma.len();

        // Extract mu and omega diagonal
        let mu: Vec<f64> = (0..n_fixed).map(|i| population.mu()[i]).collect();
        let omega_diag: Vec<f64> = (0..n_fixed).map(|i| population.omega()[(i, i)]).collect();
        let omega_sd: Vec<f64> = omega_diag.iter().map(|v| v.sqrt()).collect();

        // Coefficient of variation
        let cv_percent: Vec<f64> = mu
            .iter()
            .zip(omega_sd.iter())
            .map(|(m, s)| {
                if m.abs() > 1e-10 {
                    100.0 * s / m.abs()
                } else {
                    f64::NAN
                }
            })
            .collect();

        // Calculate shrinkage
        let pop_var = faer::Col::from_fn(n_fixed, |i| omega_diag[i]);
        let shrinkage_opt = individual_estimates.shrinkage(&pop_var);
        let eta_shrinkage: Vec<f64> = shrinkage_opt
            .map(|s| (0..s.nrows()).map(|i| s[i]).collect())
            .unwrap_or_else(|| vec![f64::NAN; n_fixed]);

        let eta_shrinkage_overall = if !eta_shrinkage.is_empty() {
            eta_shrinkage.iter().filter(|v| !v.is_nan()).sum::<f64>()
                / eta_shrinkage.iter().filter(|v| !v.is_nan()).count().max(1) as f64
        } else {
            f64::NAN
        };

        // Best likelihood for AIC/BIC
        let best_ll = ll_gq.or(ll_is).or(ll_lin).unwrap_or(-objf / 2.0);
        let best_objf = -2.0 * best_ll;

        // AIC = -2LL + 2k
        let aic = best_objf + 2.0 * n_total as f64;

        // BIC = -2LL + k * ln(n)
        let bic = best_objf + (n_total as f64) * (n_subjects as f64).ln();

        Self {
            n_subjects,
            n_observations,
            n_fixed,
            n_random,
            n_total_params: n_total,
            iterations,
            converged,
            objf,
            ll_is,
            ll_lin,
            ll_gq,
            aic,
            bic,
            eta_shrinkage,
            eta_shrinkage_overall,
            sigma,
            mu,
            omega_diag,
            omega_sd,
            cv_percent,
        }
    }

    /// Write statistics to a CSV file
    pub fn write(&self, settings: &Settings) -> Result<()> {
        let outputfile = OutputFile::new(&settings.output().path, "statistics.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(&outputfile.file);

        // Write as key-value pairs for easy reading
        writer.write_record(["metric", "value"])?;

        // Dataset info
        writer.write_record(["n_subjects", &self.n_subjects.to_string()])?;
        writer.write_record(["n_observations", &self.n_observations.to_string()])?;
        writer.write_record(["n_fixed_params", &self.n_fixed.to_string()])?;
        writer.write_record(["n_random_params", &self.n_random.to_string()])?;
        writer.write_record(["n_total_params", &self.n_total_params.to_string()])?;

        // Convergence
        writer.write_record(["iterations", &self.iterations.to_string()])?;
        writer.write_record(["converged", &self.converged.to_string()])?;

        // Likelihood
        writer.write_record(["objf", &format!("{:.6}", self.objf)])?;
        if let Some(ll) = self.ll_is {
            writer.write_record(["ll_is", &format!("{:.6}", ll)])?;
        }
        if let Some(ll) = self.ll_lin {
            writer.write_record(["ll_lin", &format!("{:.6}", ll)])?;
        }
        if let Some(ll) = self.ll_gq {
            writer.write_record(["ll_gq", &format!("{:.6}", ll)])?;
        }

        // Information criteria
        writer.write_record(["aic", &format!("{:.4}", self.aic)])?;
        writer.write_record(["bic", &format!("{:.4}", self.bic)])?;

        // Shrinkage
        writer.write_record([
            "eta_shrinkage_overall",
            &format!("{:.4}", self.eta_shrinkage_overall),
        ])?;

        // Residual error
        for (i, s) in self.sigma.iter().enumerate() {
            let key = if self.sigma.len() == 1 {
                "sigma".to_string()
            } else {
                format!("sigma_{}", i + 1)
            };
            writer.write_record([&key, &format!("{:.6}", s)])?;
        }

        writer.flush()?;
        tracing::debug!("Statistics written to {:?}", outputfile.relative_path());

        Ok(())
    }

    /// Write detailed shrinkage information
    pub fn write_shrinkage(&self, settings: &Settings, param_names: &[String]) -> Result<()> {
        let outputfile = OutputFile::new(&settings.output().path, "shrinkage.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(&outputfile.file);

        writer.write_record(["parameter", "shrinkage"])?;

        for (name, shrink) in param_names.iter().zip(self.eta_shrinkage.iter()) {
            writer.write_record([name, &format!("{:.6}", shrink)])?;
        }

        writer.flush()?;
        tracing::debug!("Shrinkage written to {:?}", outputfile.relative_path());

        Ok(())
    }
}

/// Residual error estimates
///
/// Stores the estimated residual error model parameters from SAEM
#[derive(Debug, Clone, Default, Serialize)]
pub struct ResidualErrorEstimates {
    /// Additive error component (a)
    pub additive: Option<f64>,
    /// Proportional error component (b)
    pub proportional: Option<f64>,
    /// Combined components if using combined model
    pub combined: Option<(f64, f64)>,
    /// Error model type string
    pub model_type: String,
}

impl ResidualErrorEstimates {
    /// Create from additive error only
    pub fn additive(sigma: f64) -> Self {
        Self {
            additive: Some(sigma),
            proportional: None,
            combined: None,
            model_type: "additive".to_string(),
        }
    }

    /// Create from proportional error only
    pub fn proportional(sigma: f64) -> Self {
        Self {
            additive: None,
            proportional: Some(sigma),
            combined: None,
            model_type: "proportional".to_string(),
        }
    }

    /// Create from combined error
    pub fn combined(additive: f64, proportional: f64) -> Self {
        Self {
            additive: Some(additive),
            proportional: Some(proportional),
            combined: Some((additive, proportional)),
            model_type: "combined".to_string(),
        }
    }

    /// Get all sigma values as a vector
    pub fn as_vec(&self) -> Vec<f64> {
        match (&self.additive, &self.proportional) {
            (Some(a), Some(b)) => vec![*a, *b],
            (Some(a), None) => vec![*a],
            (None, Some(b)) => vec![*b],
            (None, None) => vec![],
        }
    }

    /// Write to CSV
    pub fn write(&self, settings: &Settings) -> Result<()> {
        let outputfile = OutputFile::new(&settings.output().path, "sigma.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(&outputfile.file);

        writer.write_record(["parameter", "value", "description"])?;

        writer.write_record(["model_type", &self.model_type, ""])?;

        if let Some(a) = self.additive {
            writer.write_record(["sigma_add", &format!("{:.6}", a), "Additive error SD"])?;
        }

        if let Some(b) = self.proportional {
            writer.write_record([
                "sigma_prop",
                &format!("{:.6}", b),
                "Proportional error coefficient",
            ])?;
        }

        writer.flush()?;
        tracing::debug!("Sigma written to {:?}", outputfile.relative_path());

        Ok(())
    }
}
