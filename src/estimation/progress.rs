use serde::Serialize;

use crate::algorithms::Status;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct NonparametricCycleProgress {
    pub cycle: usize,
    pub objective: f64,
    pub objective_delta: Option<f64>,
    pub elapsed_ms: u64,
    pub status: Status,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FitProgress {
    NonparametricCycle(NonparametricCycleProgress),
}
