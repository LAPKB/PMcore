use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FitSummary {
    pub objective_function: f64,
    pub converged: bool,
    pub iterations: usize,
    pub subject_count: usize,
    pub observation_count: usize,
    pub parameter_count: usize,
    pub algorithm: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PopulationSummary {
    pub parameters: Vec<ParameterSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterSummary {
    pub name: String,
    pub mean: f64,
    pub median: f64,
    pub sd: f64,
    pub cv_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndividualSummary {
    pub id: String,
    pub parameter_names: Vec<String>,
    pub estimates: Vec<f64>,
    pub standard_errors: Option<Vec<f64>>,
}