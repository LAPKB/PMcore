use anyhow::{anyhow, bail, Result};
use pharmsol::equation::Equation;
use pharmsol::{Analytical, ValidatedModelMetadata, ODE, SDE};

pub mod covariate_model;
pub mod covariates;
pub mod metadata;
pub mod parameter_space;
pub mod variability;

pub use covariate_model::CovariateModel;
pub use covariates::{CovariateEffectsSpec, CovariateSpec};
pub use metadata::ModelMetadata;
pub use parameter_space::{
    Parameter, ParameterDomain, ParameterSpace, ParameterTransform, ParameterVariability,
};
pub use variability::{CovarianceStructure, RandomEffectsSpec, VariabilityModel};

#[derive(Debug, Clone)]
pub struct ModelDefinition<E: Equation> {
    pub equation: E,
    pub parameters: ParameterSpace,
    pub variability: VariabilityModel,
    pub covariates: CovariateSpec,
    pub metadata: ModelMetadata,
}

impl<E: Equation> ModelDefinition<E> {
    pub fn builder(equation: E) -> ModelDefinitionBuilder<E> {
        ModelDefinitionBuilder {
            equation,
            parameters: ParameterSpace::new(),
            variability: Some(VariabilityModel::default()),
            covariates: Some(CovariateSpec::InEquation),
            metadata: Some(ModelMetadata::default()),
        }
    }
}

impl<E: EquationMetadataSource> ModelDefinition<E> {
    pub fn parameter_count(&self) -> usize {
        self.equation
            .equation_metadata()
            .map_or(0, |metadata| metadata.parameters().len())
    }

    pub fn parameter_name(&self, index: usize) -> Option<&str> {
        self.equation
            .equation_metadata()
            .and_then(|metadata| metadata.parameters().get(index))
            .map(|parameter| parameter.name())
    }

    pub fn parameter_index(&self, name: &str) -> Option<usize> {
        self.equation
            .equation_metadata()
            .and_then(|metadata| metadata.parameter_index(name))
    }

    pub fn output_count(&self) -> usize {
        self.equation
            .equation_metadata()
            .map_or(0, |metadata| metadata.outputs().len())
    }

    pub fn output_name(&self, outeq: usize) -> Option<&str> {
        self.equation
            .equation_metadata()
            .and_then(|metadata| metadata.outputs().get(outeq))
            .map(|output| output.name())
    }

    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.equation.equation_metadata().and_then(|metadata| {
            metadata
                .outputs()
                .iter()
                .position(|output| output.name() == name)
        })
    }
}

pub struct ModelDefinitionBuilder<E: Equation> {
    equation: E,
    parameters: ParameterSpace,
    variability: Option<VariabilityModel>,
    covariates: Option<CovariateSpec>,
    metadata: Option<ModelMetadata>,
}

impl<E: Equation> ModelDefinitionBuilder<E> {
    pub fn parameter(mut self, parameter: Parameter) -> Result<Self>
    where
        E: EquationMetadataSource,
    {
        validate_parameter(&self.equation, &self.parameters, &parameter)?;
        self.parameters.push(parameter);
        Ok(self)
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

    pub fn build(self) -> Result<ModelDefinition<E>>
    where
        E: EquationMetadataSource,
    {
        let ModelDefinitionBuilder {
            equation,
            parameters,
            variability,
            covariates,
            metadata,
        } = self;

        if parameters.is_empty() {
            bail!("model parameters cannot be empty");
        }

        Ok(ModelDefinition {
            equation,
            parameters,
            variability: variability.unwrap_or_default(),
            covariates: covariates.unwrap_or_default(),
            metadata: metadata.unwrap_or_default(),
        })
    }
}

impl<E: EquationMetadataSource> ModelDefinitionBuilder<E> {
    pub(crate) fn output_index(&self, name: &str) -> Option<usize> {
        self.equation.equation_metadata().and_then(|metadata| {
            metadata
                .outputs()
                .iter()
                .position(|output| output.name() == name)
        })
    }
}

fn validate_parameter<E: EquationMetadataSource>(
    equation: &E,
    existing: &ParameterSpace,
    parameter: &Parameter,
) -> Result<()> {
    let metadata = equation
        .equation_metadata()
        .ok_or_else(|| anyhow!("equation metadata is required to define model parameters"))?;

    if parameter.name.trim().is_empty() {
        bail!("model parameter name cannot be empty");
    }

    if metadata.parameter_index(&parameter.name).is_none() {
        bail!("unknown equation parameter: {}", parameter.name);
    }

    if existing
        .iter()
        .any(|existing_parameter| existing_parameter.name == parameter.name)
    {
        bail!("duplicate model parameter: {}", parameter.name);
    }

    Ok(())
}

pub trait EquationMetadataSource: Equation {
    fn equation_metadata(&self) -> Option<&ValidatedModelMetadata>;
}

impl EquationMetadataSource for ODE {
    fn equation_metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.metadata()
    }
}

impl EquationMetadataSource for Analytical {
    fn equation_metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.metadata()
    }
}

impl EquationMetadataSource for SDE {
    fn equation_metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.metadata()
    }
}

#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load"),
    all(
        feature = "dsl-wasm",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    )
))]
impl EquationMetadataSource for pharmsol::dsl::RuntimeOdeModel {
    fn equation_metadata(&self) -> Option<&ValidatedModelMetadata> {
        Some(self.metadata())
    }
}

#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load"),
    all(
        feature = "dsl-wasm",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    )
))]
impl EquationMetadataSource for pharmsol::dsl::RuntimeAnalyticalModel {
    fn equation_metadata(&self) -> Option<&ValidatedModelMetadata> {
        Some(self.metadata())
    }
}

#[cfg(any(
    feature = "dsl-jit",
    all(feature = "dsl-aot", feature = "dsl-aot-load"),
    all(
        feature = "dsl-wasm",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    )
))]
impl EquationMetadataSource for pharmsol::dsl::RuntimeSdeModel {
    fn equation_metadata(&self) -> Option<&ValidatedModelMetadata> {
        Some(self.metadata())
    }
}
