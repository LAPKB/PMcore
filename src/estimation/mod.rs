pub mod error_models;
pub mod nonparametric;
pub mod problem;
pub mod progress;

pub use crate::algorithms::nonparametric::{NpagConfig, NpmapConfig, NpodConfig};
pub use error_models::ErrorModels;
pub use problem::{
    EstimationProblem, Framework, NonParametric, Parametric, ParametricBuilder,
};
pub use progress::{FitProgress, NonparametricCycleProgress};
