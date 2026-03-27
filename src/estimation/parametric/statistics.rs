//! Statistical summaries for parametric algorithm results.

use anyhow::Result;
use csv::WriterBuilder;
use pharmsol::{ResidualErrorModel, ResidualErrorModels};
use serde::Serialize;

use crate::estimation::parametric::{IndividualEstimates, Population};
use crate::output::OutputFile;

#[derive(Debug, Clone, Default, Serialize)]
pub struct ParametricStatistics {
    pub n_subjects: usize,
    pub n_observations: usize,
    pub n_fixed: usize,
    pub n_random: usize,
    pub n_total_params: usize,
    pub iterations: usize,
    pub converged: bool,
    pub objf: f64,
    pub ll_is: Option<f64>,
    pub ll_lin: Option<f64>,
    pub ll_gq: Option<f64>,
    pub aic: f64,
    pub bic: f64,
    pub eta_shrinkage: Vec<f64>,
    pub eta_shrinkage_overall: f64,
    pub sigma: Vec<f64>,
    pub mu: Vec<f64>,
    pub omega_diag: Vec<f64>,
    pub omega_sd: Vec<f64>,
    pub cv_percent: Vec<f64>,
}

impl ParametricStatistics {
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
        let n_random = n_fixed;
        let n_total = n_fixed + n_random + sigma.len();

        let mu: Vec<f64> = (0..n_fixed).map(|i| population.mu()[i]).collect();
        let omega_diag: Vec<f64> = (0..n_fixed).map(|i| population.omega()[(i, i)]).collect();
        let omega_sd: Vec<f64> = omega_diag.iter().map(|v| v.sqrt()).collect();
        let cv_percent: Vec<f64> = population
            .coefficient_of_variation()
            .iter()
            .copied()
            .collect();

        let pop_var = faer::Col::from_fn(n_fixed, |i| omega_diag[i]);
        let shrinkage_opt = individual_estimates.shrinkage(&pop_var);
        let eta_shrinkage: Vec<f64> = shrinkage_opt
            .map(|shrinkage| (0..shrinkage.nrows()).map(|i| shrinkage[i]).collect())
            .unwrap_or_else(|| vec![f64::NAN; n_fixed]);

        let eta_shrinkage_overall = if !eta_shrinkage.is_empty() {
            eta_shrinkage.iter().filter(|v| !v.is_nan()).sum::<f64>()
                / eta_shrinkage.iter().filter(|v| !v.is_nan()).count().max(1) as f64
        } else {
            f64::NAN
        };

        let best_ll = ll_gq.or(ll_is).or(ll_lin).unwrap_or(-objf / 2.0);
        let best_objf = -2.0 * best_ll;
        let aic = best_objf + 2.0 * n_total as f64;
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

    pub fn write(&self, folder: &str) -> Result<()> {
        let outputfile = OutputFile::new(folder, "statistics.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(outputfile.file());

        writer.write_record(["metric", "value"])?;
        writer.write_record(["n_subjects", &self.n_subjects.to_string()])?;
        writer.write_record(["n_observations", &self.n_observations.to_string()])?;
        writer.write_record(["n_fixed_params", &self.n_fixed.to_string()])?;
        writer.write_record(["n_random_params", &self.n_random.to_string()])?;
        writer.write_record(["n_total_params", &self.n_total_params.to_string()])?;
        writer.write_record(["iterations", &self.iterations.to_string()])?;
        writer.write_record(["converged", &self.converged.to_string()])?;
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
        writer.write_record(["aic", &format!("{:.4}", self.aic)])?;
        writer.write_record(["bic", &format!("{:.4}", self.bic)])?;
        writer.write_record([
            "eta_shrinkage_overall",
            &format!("{:.4}", self.eta_shrinkage_overall),
        ])?;

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

    pub fn write_shrinkage(&self, folder: &str, param_names: &[String]) -> Result<()> {
        let outputfile = OutputFile::new(folder, "shrinkage.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(outputfile.file());

        writer.write_record(["parameter", "shrinkage"])?;

        for (name, shrink) in param_names.iter().zip(self.eta_shrinkage.iter()) {
            writer.write_record([name, &format!("{:.6}", shrink)])?;
        }

        writer.flush()?;
        tracing::debug!("Shrinkage written to {:?}", outputfile.relative_path());
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ResidualErrorEstimates {
    pub additive: Option<f64>,
    pub proportional: Option<f64>,
    pub combined: Option<(f64, f64)>,
    pub model_type: String,
}

impl ResidualErrorEstimates {
    pub fn additive(sigma: f64) -> Self {
        Self {
            additive: Some(sigma),
            proportional: None,
            combined: None,
            model_type: "additive".to_string(),
        }
    }

    pub fn proportional(sigma: f64) -> Self {
        Self {
            additive: None,
            proportional: Some(sigma),
            combined: None,
            model_type: "proportional".to_string(),
        }
    }

    pub fn combined(additive: f64, proportional: f64) -> Self {
        Self {
            additive: Some(additive),
            proportional: Some(proportional),
            combined: Some((additive, proportional)),
            model_type: "combined".to_string(),
        }
    }

    pub fn as_vec(&self) -> Vec<f64> {
        match (&self.additive, &self.proportional) {
            (Some(a), Some(b)) => vec![*a, *b],
            (Some(a), None) => vec![*a],
            (None, Some(b)) => vec![*b],
            (None, None) => vec![],
        }
    }

    pub fn write(&self, folder: &str) -> Result<()> {
        let outputfile = OutputFile::new(folder, "residual_error.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(outputfile.file());

        writer.write_record(["parameter", "value", "description"])?;
        writer.write_record(["model_type", &self.model_type, ""])?;

        if let Some(additive) = self.additive {
            writer.write_record([
                "sigma_add",
                &format!("{:.6}", additive),
                "Additive error SD",
            ])?;
        }

        if let Some(proportional) = self.proportional {
            writer.write_record([
                "sigma_prop",
                &format!("{:.6}", proportional),
                "Proportional error coefficient",
            ])?;
        }

        writer.flush()?;
        tracing::debug!("Sigma written to {:?}", outputfile.relative_path());
        Ok(())
    }
}

pub fn residual_error_estimates_from_models(
    error_models: &ResidualErrorModels,
) -> ResidualErrorEstimates {
    let models = error_models
        .iter()
        .map(|(_, model)| *model)
        .collect::<Vec<_>>();

    let Some(first) = models.first().copied() else {
        return ResidualErrorEstimates::default();
    };

    if !models.iter().all(|model| *model == first) {
        return ResidualErrorEstimates::default();
    }

    match first {
        ResidualErrorModel::Constant { a } => ResidualErrorEstimates::additive(a),
        ResidualErrorModel::Proportional { b } => ResidualErrorEstimates::proportional(b),
        ResidualErrorModel::Combined { a, b } => ResidualErrorEstimates::combined(a, b),
        ResidualErrorModel::Exponential { .. } => ResidualErrorEstimates {
            model_type: "exponential".to_string(),
            ..ResidualErrorEstimates::default()
        },
    }
}

pub fn residual_error_estimates_from_observed_outeqs(
    error_models: &ResidualErrorModels,
    observed_outeqs: &[usize],
) -> ResidualErrorEstimates {
    let models = error_models
        .iter()
        .filter(|(outeq, _)| observed_outeqs.contains(outeq))
        .map(|(_, model)| *model)
        .collect::<Vec<_>>();

    let Some(first) = models.first().copied() else {
        return ResidualErrorEstimates::default();
    };

    if !models.iter().all(|model| *model == first) {
        return ResidualErrorEstimates::default();
    }

    match first {
        ResidualErrorModel::Constant { a } => ResidualErrorEstimates::additive(a),
        ResidualErrorModel::Proportional { b } => ResidualErrorEstimates::proportional(b),
        ResidualErrorModel::Combined { a, b } => ResidualErrorEstimates::combined(a, b),
        ResidualErrorModel::Exponential { .. } => ResidualErrorEstimates {
            model_type: "exponential".to_string(),
            ..ResidualErrorEstimates::default()
        },
    }
}
