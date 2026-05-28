use anyhow::{anyhow, bail, Result};
use pharmsol::equation::Equation;
use pharmsol::{Analytical, ValidatedModelMetadata, ODE, SDE};

pub mod metadata;
pub mod parameter_space;
pub mod variability;

pub use metadata::ModelMetadata;
pub use parameter_space::{
    Parameter, ParameterDomain, ParameterSpace, ParameterTransform, ParameterVariability,
};
pub use variability::{CovarianceStructure, RandomEffectsSpec, VariabilityModel};

#[derive(Debug, Clone)]
pub struct Model<E: Equation> {
    pub equation: E,
    pub parameters: ParameterSpace,
}

impl<E: Equation> Model<E> {
    pub fn builder(equation: E) -> ModelBuilder<E> {
        ModelBuilder {
            equation,
            parameters: ParameterSpace::new(),
        }
    }
}

impl<E: EquationMetadataSource> Model<E> {
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

pub struct ModelBuilder<E: Equation> {
    equation: E,
    parameters: ParameterSpace,
}

impl<E: Equation> ModelBuilder<E> {
    pub fn parameter(mut self, parameter: Parameter) -> Result<Self>
    where
        E: EquationMetadataSource,
    {
        validate_parameter(&self.equation, &self.parameters, &parameter)?;
        self.parameters.push(parameter);
        Ok(self)
    }

    pub fn build(self) -> Result<Model<E>>
    where
        E: EquationMetadataSource,
    {
        let ModelBuilder {
            equation,
            parameters,
        } = self;

        if parameters.is_empty() {
            bail!("model parameters cannot be empty");
        }

        Ok(Model {
            equation,
            parameters,
        })
    }
}

impl<E: EquationMetadataSource> ModelBuilder<E> {
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
