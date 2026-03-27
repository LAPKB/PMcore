use anyhow::Result;
use csv::WriterBuilder;
use faer::{Col, Mat};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};

use crate::algorithms::Status;
use crate::estimation::parametric::Population;
use crate::output::OutputFile;

#[derive(Debug, Clone, Default, Serialize)]
pub struct LikelihoodEstimates {
    pub ll_linearization: Option<f64>,
    pub ll_importance_sampling: Option<f64>,
    pub ll_gaussian_quadrature: Option<f64>,
    pub is_n_samples: Option<usize>,
    pub gq_n_points: Option<usize>,
}

impl LikelihoodEstimates {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn best_estimate(&self) -> Option<f64> {
        self.ll_gaussian_quadrature
            .or(self.ll_importance_sampling)
            .or(self.ll_linearization)
    }

    pub fn best_objf(&self) -> Option<f64> {
        self.best_estimate().map(|ll| -2.0 * ll)
    }
}

#[derive(Debug, Clone, Default)]
pub struct UncertaintyEstimates {
    pub fim: Option<Mat<f64>>,
    pub fim_inverse: Option<Mat<f64>>,
    pub se_mu: Option<Col<f64>>,
    pub se_omega: Option<Mat<f64>>,
    pub rse_mu: Option<Col<f64>>,
    pub fim_method: Option<FimMethod>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FimMethod {
    Observed,
    Expected,
    StochasticApproximation,
    Linearization,
}

#[derive(Debug, Clone, Default)]
pub struct ParametricIterationLog {
    iterations: Vec<usize>,
    objf: Vec<f64>,
    mu_history: Vec<Vec<f64>>,
    omega_diag_history: Vec<Vec<f64>>,
    status: Vec<String>,
}

impl ParametricIterationLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn log_iteration(
        &mut self,
        iteration: usize,
        objf: f64,
        population: &Population,
        status: &Status,
    ) {
        self.iterations.push(iteration);
        self.objf.push(objf);
        self.mu_history
            .push((0..population.npar()).map(|i| population.mu()[i]).collect());
        self.omega_diag_history.push(
            (0..population.npar())
                .map(|i| population.omega()[(i, i)])
                .collect(),
        );
        self.status.push(format!("{:?}", status));
    }

    pub fn len(&self) -> usize {
        self.iterations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.iterations.is_empty()
    }

    pub fn objf_history(&self) -> &[f64] {
        &self.objf
    }

    pub fn write(&self, folder: &str, param_names: &[String]) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }

        let outputfile = OutputFile::new(folder, "iterations.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(outputfile.file());

        let n_params = self
            .mu_history
            .first()
            .map(|values| values.len())
            .unwrap_or(0);
        let mut header = vec!["iteration".to_string(), "objf".to_string()];
        for name in param_names {
            header.push(format!("mu_{}", name));
        }
        for name in param_names {
            header.push(format!("omega_{}", name));
        }
        header.push("status".to_string());
        writer.write_record(&header)?;

        for index in 0..self.iterations.len() {
            let mut row = vec![
                self.iterations[index].to_string(),
                format!("{:.6}", self.objf[index]),
            ];

            for parameter_index in 0..n_params {
                row.push(format!(
                    "{:.6}",
                    self.mu_history[index].get(parameter_index).unwrap_or(&0.0)
                ));
            }

            for parameter_index in 0..n_params {
                row.push(format!(
                    "{:.6}",
                    self.omega_diag_history[index]
                        .get(parameter_index)
                        .unwrap_or(&0.0)
                ));
            }

            row.push(self.status[index].clone());
            writer.write_record(&row)?;
        }

        writer.flush()?;
        Ok(())
    }
}

impl Serialize for ParametricIterationLog {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ParametricIterationLog", 5)?;
        state.serialize_field("iterations", &self.iterations)?;
        state.serialize_field("objf", &self.objf)?;
        state.serialize_field("mu_history", &self.mu_history)?;
        state.serialize_field("omega_diag_history", &self.omega_diag_history)?;
        state.serialize_field("status", &self.status)?;
        state.end()
    }
}
