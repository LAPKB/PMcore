use anyhow::Result;
use pharmsol::AssayErrorModel;
use serde::Serialize;

use crate::algorithms::Status;
use crate::estimation::nonparametric::{median, NPCycle};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FitControl {
    PauseAfterCycle,
    Resume,
    StopAfterCycle,
    Ping,
}

pub trait FitControlSource {
    fn next_control(&mut self) -> Result<Option<FitControl>>;
}

impl<F> FitControlSource for F
where
    F: FnMut() -> Result<Option<FitControl>>,
{
    fn next_control(&mut self) -> Result<Option<FitControl>> {
        self()
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NonparametricErrorModelKind {
    Gamma,
    Lambda,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct NonparametricErrorModelProgress {
    pub outeq: usize,
    pub kind: NonparametricErrorModelKind,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct NonparametricParameterProgress {
    pub name: String,
    pub mean: f64,
    pub median: f64,
    pub sd: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct NonparametricCycleProgress {
    pub cycle: usize,
    pub neg2ll: f64,
    pub objective_delta: Option<f64>,
    pub cycle_elapsed_ms: u64,
    pub total_elapsed_ms: u64,
    pub nspp: usize,
    pub status: Status,
    pub error_models: Vec<NonparametricErrorModelProgress>,
    pub parameters: Vec<NonparametricParameterProgress>,
}

impl NonparametricCycleProgress {
    pub(crate) fn from_cycle(
        cycle: &NPCycle,
        parameter_names: &[String],
        objective_delta: Option<f64>,
        cycle_elapsed_ms: u64,
        total_elapsed_ms: u64,
    ) -> Result<Self> {
        let error_models = cycle
            .error_models()
            .iter()
            .map(|(outeq, error_model)| match error_model {
                AssayErrorModel::Additive { .. } => Ok(Some(NonparametricErrorModelProgress {
                    outeq,
                    kind: NonparametricErrorModelKind::Lambda,
                    value: error_model.factor()?,
                })),
                AssayErrorModel::Proportional { .. } => Ok(Some(NonparametricErrorModelProgress {
                    outeq,
                    kind: NonparametricErrorModelKind::Gamma,
                    value: error_model.factor()?,
                })),
                AssayErrorModel::None => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        let parameters = cycle
            .theta()
            .matrix()
            .col_iter()
            .enumerate()
            .map(|(index, values)| {
                let parameter_values: Vec<f64> = values.iter().copied().collect();
                let mean = parameter_values.iter().sum::<f64>() / parameter_values.len() as f64;
                let variance = parameter_values
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / (parameter_values.len() as f64 - 1.0);

                NonparametricParameterProgress {
                    name: parameter_names
                        .get(index)
                        .cloned()
                        .unwrap_or_else(|| format!("parameter_{}", index + 1)),
                    mean,
                    median: median(&parameter_values),
                    sd: variance.sqrt(),
                }
            })
            .collect();

        Ok(Self {
            cycle: cycle.cycle(),
            neg2ll: cycle.objf(),
            objective_delta,
            cycle_elapsed_ms,
            total_elapsed_ms,
            nspp: cycle.nspp(),
            status: cycle.status().clone(),
            error_models,
            parameters,
        })
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FitProgress {
    FitStarted,
    NonparametricCycle(NonparametricCycleProgress),
    Paused { cycle: usize },
    Resumed { cycle: usize },
    StopRequested { cycle: usize },
    FitCompleted { cycles: usize, status: Status },
}
