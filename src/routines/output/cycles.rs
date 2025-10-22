use anyhow::Result;
use csv::WriterBuilder;
use pharmsol::{ErrorModel, ErrorModels};
use serde::Serialize;

use crate::{
    algorithms::Status,
    prelude::Settings,
    routines::output::{median, OutputFile},
    structs::theta::Theta,
};

/// An [NPCycle] object contains the summary of a cycle
/// It holds the following information:
/// - `cycle`: The cycle number
/// - `objf`: The objective function value
/// - `gamlam`: The assay noise parameter, either gamma or lambda
/// - `theta`: The support points and their associated probabilities
/// - `nspp`: The number of support points
/// - `delta_objf`: The change in objective function value from last cycle
/// - `converged`: Whether the algorithm has reached convergence
#[derive(Debug, Clone, Serialize)]
pub struct NPCycle {
    cycle: usize,
    objf: f64,
    error_models: ErrorModels,
    theta: Theta,
    nspp: usize,
    delta_objf: f64,
    status: Status,
}

impl NPCycle {
    pub fn new(
        cycle: usize,
        objf: f64,
        error_models: ErrorModels,
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
    pub fn error_models(&self) -> &ErrorModels {
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
            error_models: ErrorModels::default(),
            theta: Theta::new(),
            nspp: 0,
            delta_objf: 0.0,
            status: Status::Starting,
        }
    }
}

/// This holdes a vector of [NPCycle] objects to provide a more detailed log
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

    pub fn write(&self, settings: &Settings) -> Result<()> {
        tracing::debug!("Writing cycles...");
        let outputfile = OutputFile::new(&settings.output().path, "cycles.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(false)
            .from_writer(&outputfile.file);

        // Write headers
        writer.write_field("cycle")?;
        writer.write_field("converged")?;
        writer.write_field("status")?;
        writer.write_field("neg2ll")?;
        writer.write_field("nspp")?;
        if let Some(first_cycle) = self.cycles.first() {
            first_cycle.error_models.iter().try_for_each(
                |(outeq, errmod): (usize, &ErrorModel)| -> Result<(), csv::Error> {
                    match errmod {
                        ErrorModel::Additive { .. } => {
                            writer.write_field(format!("gamlam.{}", outeq))?;
                        }
                        ErrorModel::Proportional { .. } => {
                            writer.write_field(format!("gamlam.{}", outeq))?;
                        }
                        ErrorModel::None => {}
                    }
                    Ok(())
                },
            )?;
        }

        let parameter_names = settings.parameters().names();
        for param_name in &parameter_names {
            writer.write_field(format!("{}.mean", param_name))?;
            writer.write_field(format!("{}.median", param_name))?;
            writer.write_field(format!("{}.sd", param_name))?;
        }

        writer.write_record(None::<&[u8]>)?;

        for cycle in &self.cycles {
            writer.write_field(format!("{}", cycle.cycle))?;
            writer.write_field(format!("{}", cycle.status == Status::Converged))?;
            writer.write_field(format!("{}", cycle.status))?;
            writer.write_field(format!("{}", cycle.objf))?;
            writer
                .write_field(format!("{}", cycle.theta.nspp()))
                .unwrap();

            // Write the error models
            cycle.error_models.iter().try_for_each(
                |(_, errmod): (usize, &ErrorModel)| -> Result<()> {
                    match errmod {
                        ErrorModel::Additive { lambda: _, poly: _ } => {
                            writer.write_field(format!("{:.5}", errmod.factor()?))?;
                        }
                        ErrorModel::Proportional { gamma: _, poly: _ } => {
                            writer.write_field(format!("{:.5}", errmod.factor()?))?;
                        }
                        ErrorModel::None => {}
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
