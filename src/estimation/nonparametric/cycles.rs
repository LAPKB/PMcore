use std::{fs::File, path::Path};

use anyhow::Result;
use csv::WriterBuilder;
use pharmsol::{AssayErrorModel, AssayErrorModels};
use serde::Serialize;

use crate::{
    algorithms::Status,
    estimation::nonparametric::{median, theta::Theta, weights::Weights},
};

/// A snapshot of the fit state after one cycle.
///
/// The objective function is stored on the scale that is actually minimized,
/// `-2 × log-likelihood`, so that cycles logged by different algorithms are
/// directly comparable.
#[derive(Debug, Clone, Serialize)]
pub struct NPCycle {
    cycle: usize,
    n2ll: f64,
    error_models: AssayErrorModels,
    theta: Theta,
    weights: Weights,
    nspp: usize,
    delta_log_likelihood: f64,
    status: Status,
}

impl NPCycle {
    pub fn new(
        cycle: usize,
        n2ll: f64,
        error_models: AssayErrorModels,
        theta: Theta,
        weights: Weights,
        delta_log_likelihood: f64,
        status: Status,
    ) -> Self {
        let nspp = theta.nspp();

        Self {
            cycle,
            n2ll,
            error_models,
            theta,
            weights,
            nspp,
            delta_log_likelihood,
            status,
        }
    }

    pub fn cycle(&self) -> usize {
        self.cycle
    }
    /// The objective function minimized by the algorithm, `-2 × log-likelihood`.
    pub fn n2ll(&self) -> f64 {
        self.n2ll
    }
    pub fn error_models(&self) -> &AssayErrorModels {
        &self.error_models
    }
    pub fn theta(&self) -> &Theta {
        &self.theta
    }
    pub fn weights(&self) -> &Weights {
        &self.weights
    }
    pub fn nspp(&self) -> usize {
        self.nspp
    }
    /// Change in log-likelihood since the previous cycle.
    pub fn delta_log_likelihood(&self) -> f64 {
        self.delta_log_likelihood
    }
    pub fn status(&self) -> &Status {
        &self.status
    }

    pub fn placeholder() -> Self {
        Self {
            cycle: 0,
            n2ll: 0.0,
            error_models: AssayErrorModels::default(),
            theta: Theta::new(),
            weights: Weights::default(),
            nspp: 0,
            delta_log_likelihood: 0.0,
            status: Status::Continue,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CycleLog {
    cycles: Vec<NPCycle>,
}

impl CycleLog {
    pub fn new() -> Self {
        Self { cycles: Vec::new() }
    }

    pub fn cycles(&self) -> &[NPCycle] {
        &self.cycles
    }

    pub fn push(&mut self, cycle: NPCycle) {
        self.cycles.push(cycle);
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        tracing::debug!("Writing cycles...");

        super::create_parent_dir(path)?;
        let file = File::create(path)?;
        let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

        writer.write_field("cycle")?;
        writer.write_field("converged")?;
        writer.write_field("status")?;
        writer.write_field("neg2ll")?;
        writer.write_field("nspp")?;
        if let Some(first_cycle) = self.cycles.first() {
            first_cycle.error_models.iter().try_for_each(
                |(outeq, errmod): (usize, &AssayErrorModel)| -> Result<(), csv::Error> {
                    match errmod {
                        AssayErrorModel::Additive { .. } => {
                            writer.write_field(format!("gamlam.{}", outeq))?;
                        }
                        AssayErrorModel::Proportional { .. } => {
                            writer.write_field(format!("gamlam.{}", outeq))?;
                        }
                        AssayErrorModel::None => {}
                    }
                    Ok(())
                },
            )?;
        }

        let names = self
            .cycles
            .first()
            .map(|cycle| cycle.theta.param_names())
            .expect("No cycles");

        for param_name in names {
            writer.write_field(format!("{}.mean", param_name))?;
            writer.write_field(format!("{}.median", param_name))?;
            writer.write_field(format!("{}.sd", param_name))?;
        }

        writer.write_record(None::<&[u8]>)?;

        for cycle in &self.cycles {
            writer.write_field(format!("{}", cycle.cycle))?;
            writer.write_field(format!("{}", cycle.status.converged()))?;
            writer.write_field(format!("{}", cycle.status))?;
            writer.write_field(format!("{}", cycle.n2ll))?;
            writer.write_field(format!("{}", cycle.nspp))?;

            cycle.error_models.iter().try_for_each(
                |(_, errmod): (usize, &AssayErrorModel)| -> Result<()> {
                    match errmod {
                        AssayErrorModel::Additive { lambda: _, poly: _ } => {
                            writer.write_field(format!("{:.5}", errmod.factor()?))?;
                        }
                        AssayErrorModel::Proportional { gamma: _, poly: _ } => {
                            writer.write_field(format!("{:.5}", errmod.factor()?))?;
                        }
                        AssayErrorModel::None => {}
                    }
                    Ok(())
                },
            )?;

            for param in cycle.theta.matrix().col_iter() {
                let param_values: Vec<f64> = param.iter().cloned().collect();

                let mean: f64 = param_values.iter().sum::<f64>() / param_values.len() as f64;
                let median = median(&param_values);
                let std = param_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (param_values.len() as f64 - 1.0);

                writer.write_field(format!("{}", mean))?;
                writer.write_field(format!("{}", median))?;
                writer.write_field(format!("{}", std))?;
            }
            writer.write_record(None::<&[u8]>)?;
        }
        writer.flush()?;

        tracing::debug!("Cycles written to {:?}", path);
        Ok(())
    }
}

impl Default for CycleLog {
    fn default() -> Self {
        Self::new()
    }
}
