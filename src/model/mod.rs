use anyhow::{bail, Result};
use pharmsol::equation::Equation;

pub mod covariate_model;
pub mod covariates;
pub mod metadata;
pub mod observation_spec;
pub mod parameter_space;
pub mod variability;

pub use covariate_model::CovariateModel;
pub use covariates::{CovariateEffectsSpec, CovariateSpec};
pub use metadata::ModelMetadata;
pub use observation_spec::{
    ContinuousObservationSpec, ObservationChannel, ObservationLikelihood, ObservationSpec,
};
pub use parameter_space::{
    ParameterDomain, ParameterSpace, ParameterSpec, ParameterTransform, ParameterVariability,
};
pub use variability::{CovarianceStructure, RandomEffectsSpec, VariabilityModel};

#[derive(Debug, Clone)]
pub struct ModelDefinition<E: Equation> {
    pub equation: E,
    pub parameters: ParameterSpace,
    pub observations: ObservationSpec,
    pub variability: VariabilityModel,
    pub covariates: CovariateSpec,
    pub metadata: ModelMetadata,
}

impl<E: Equation> ModelDefinition<E> {
    pub fn builder(equation: E) -> ModelDefinitionBuilder<E> {
        ModelDefinitionBuilder {
            equation,
            parameters: None,
            observations: None,
            variability: Some(VariabilityModel::default()),
            covariates: Some(CovariateSpec::InEquation),
            metadata: Some(ModelMetadata::default()),
        }
    }
}

pub struct ModelDefinitionBuilder<E: Equation> {
    equation: E,
    parameters: Option<ParameterSpace>,
    observations: Option<ObservationSpec>,
    variability: Option<VariabilityModel>,
    covariates: Option<CovariateSpec>,
    metadata: Option<ModelMetadata>,
}

impl<E: Equation> ModelDefinitionBuilder<E> {
    pub fn parameters(mut self, parameters: ParameterSpace) -> Self {
        self.parameters = Some(parameters);
        self
    }

    pub fn observations(mut self, observations: ObservationSpec) -> Self {
        self.observations = Some(observations);
        self
    }

    pub fn variability(mut self, variability: VariabilityModel) -> Self {
        self.variability = Some(variability);
        self
    }

    pub fn covariates(mut self, covariates: CovariateSpec) -> Self {
        self.covariates = Some(covariates);
        self
    }

    pub fn metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn build(self) -> Result<ModelDefinition<E>> {
        let parameters = self
            .parameters
            .ok_or_else(|| anyhow::anyhow!("model parameters are required"))?;
        if parameters.is_empty() {
            bail!("model parameters cannot be empty");
        }

        Ok(ModelDefinition {
            equation: self.equation,
            parameters,
            observations: self.observations.unwrap_or_default(),
            variability: self.variability.unwrap_or_default(),
            covariates: self.covariates.unwrap_or_default(),
            metadata: self.metadata.unwrap_or_default(),
        })
    }
}
