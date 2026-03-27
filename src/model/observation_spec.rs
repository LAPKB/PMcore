use pharmsol::prelude::data::{AssayErrorModels, ResidualErrorModels};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationSpec {
    pub channels: Vec<ObservationChannel>,
    pub assay_error_models: AssayErrorModels,
    pub residual_error_models: Option<ResidualErrorModels>,
}

impl ObservationSpec {
    pub fn new() -> Self {
        Self {
            channels: Vec::new(),
            assay_error_models: AssayErrorModels::new(),
            residual_error_models: None,
        }
    }

    pub fn add_channel(mut self, channel: ObservationChannel) -> Self {
        self.channels.push(channel);
        self
    }

    pub fn with_assay_error_models(mut self, assay_error_models: AssayErrorModels) -> Self {
        self.assay_error_models = assay_error_models;
        self
    }

    pub fn with_residual_error_models(
        mut self,
        residual_error_models: ResidualErrorModels,
    ) -> Self {
        self.residual_error_models = Some(residual_error_models);
        self
    }
}

impl Default for ObservationSpec {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObservationChannel {
    pub outeq: usize,
    pub name: String,
    pub likelihood: ObservationLikelihood,
}

impl ObservationChannel {
    pub fn continuous(outeq: usize, name: impl Into<String>) -> Self {
        Self {
            outeq,
            name: name.into(),
            likelihood: ObservationLikelihood::Continuous(ContinuousObservationSpec::default()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObservationLikelihood {
    Continuous(ContinuousObservationSpec),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContinuousObservationSpec {
    pub supports_censoring: bool,
}
