use anyhow::Result;
use csv::WriterBuilder;
use pharmsol::{AssayErrorModel, AssayErrorModels};
use serde::Serialize;

use crate::{
    algorithms::{Status, StopReason},
    estimation::nonparametric::median,
    estimation::nonparametric::theta::Theta,
    output::OutputFile,
};

#[derive(Debug, Clone, Serialize)]
pub struct NPCycle {
    cycle: usize,
    objf: f64,
    error_models: AssayErrorModels,
    theta: Theta,
    nspp: usize,
    delta_objf: f64,
    status: Status,
}

impl NPCycle {
    pub fn new(
        cycle: usize,
        objf: f64,
        error_models: AssayErrorModels,
        theta: Theta,
        nspp: usize,
        delta_objf: f64,
        status: Status,
    ) -> Self {
        Self {
            cycle,
            objf,
            error_models,
            theta,
            nspp,
            delta_objf,
            status,
        }
    }

    pub fn cycle(&self) -> usize {
        self.cycle
    }
    pub fn objf(&self) -> f64 {
        self.objf
    }
    pub fn error_models(&self) -> &AssayErrorModels {
        &self.error_models
    }
    pub fn theta(&self) -> &Theta {
        &self.theta
    }
    pub fn nspp(&self) -> usize {
        self.nspp
    }
    pub fn delta_objf(&self) -> f64 {
        self.delta_objf
    }
    pub fn status(&self) -> &Status {
        &self.status
    }

    pub fn placeholder() -> Self {
        Self {
            cycle: 0,
            objf: 0.0,
            error_models: AssayErrorModels::default(),
            theta: Theta::new(),
            nspp: 0,
            delta_objf: 0.0,
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

    pub fn write(&self, folder: &str, parameter_names: &[String]) -> Result<()> {
        tracing::debug!("Writing cycles...");
        let outputfile = OutputFile::new(folder, "iterations.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(false)
            .from_writer(outputfile.file());

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

        for param_name in parameter_names {
            writer.write_field(format!("{}.mean", param_name))?;
            writer.write_field(format!("{}.median", param_name))?;
            writer.write_field(format!("{}.sd", param_name))?;
        }

        writer.write_record(None::<&[u8]>)?;

        for cycle in &self.cycles {
            writer.write_field(format!("{}", cycle.cycle))?;
            writer.write_field(format!(
                "{}",
                cycle.status == Status::Stop(StopReason::Converged)
            ))?;
            writer.write_field(format!("{}", cycle.status))?;
            writer.write_field(format!("{}", cycle.objf))?;
            writer
                .write_field(format!("{}", cycle.theta.nspp()))
                .unwrap();

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
        tracing::debug!("Cycles written to {:?}", &outputfile.relative_path());
        Ok(())
    }
}

impl Default for CycleLog {
    fn default() -> Self {
        Self::new()
    }
}
