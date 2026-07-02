use anyhow::Result;
use pharmsol::equation::Equation;
use pharmsol::{Analytical, ValidatedModelMetadata, ODE, SDE};

pub mod metadata;
pub mod parameter_space;

pub use metadata::ModelMetadata;
// Re-exporting the new typestate parameter elements for easy access downstream
pub use parameter_space::{
    BoundedParameter, Parameter, ParameterMeta, ParameterScale, ParameterSpace, UnboundedParameter,
};

#[derive(Debug, Clone)]
pub struct Model<E: Equation> {
    pub equation: E,
    // Note: No 'parameters' field here! It is now managed by EstimationProblem<E, F>.
    // This struct is ready to hold Covariates or Variability specs if you add them later.
}

impl<E: Equation> Model<E> {
    pub fn builder(equation: E) -> ModelBuilder<E> {
        ModelBuilder { equation }
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
}

impl<E: Equation> ModelBuilder<E> {
    pub fn build(self) -> Result<Model<E>> {
        Ok(Model {
            equation: self.equation,
        })
    }
}

impl<E: EquationMetadataSource> ModelBuilder<E> {
    pub(crate) fn parameter_index(&self, name: &str) -> Option<usize> {
        self.equation
            .equation_metadata()
            .and_then(|metadata| metadata.parameter_index(name))
    }

    pub(crate) fn parameter_names(&self) -> Vec<String> {
        self.equation
            .equation_metadata()
            .map(|metadata| {
                metadata
                    .parameters()
                    .iter()
                    .map(|parameter| parameter.name().to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(crate) fn output_names(&self) -> Vec<String> {
        self.equation
            .equation_metadata()
            .map(|metadata| {
                metadata
                    .outputs()
                    .iter()
                    .map(|output| output.name().to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(crate) fn output_index(&self, name: &str) -> Option<usize> {
        self.equation.equation_metadata().and_then(|metadata| {
            metadata
                .outputs()
                .iter()
                .position(|output| output.name() == name)
        })
    }
}

pub trait EquationMetadataSource: Equation {
    fn equation_metadata(&self) -> Option<&ValidatedModelMetadata>;
}

// Macro for standard pharmsol equations
macro_rules! impl_metadata_opt {
    ($($t:ty),+) => {
        $(impl EquationMetadataSource for $t {
            fn equation_metadata(&self) -> Option<&ValidatedModelMetadata> {
                self.metadata()
            }
        })+
    };
}
impl_metadata_opt!(ODE, Analytical, SDE);

// Macro for runtime/JIT models
macro_rules! impl_metadata_some {
    ($($t:ty),+) => {
        $(
            #[cfg(any(
                feature = "dsl-jit",
                all(feature = "dsl-aot", feature = "dsl-aot-load"),
                all(feature = "dsl-wasm", not(all(target_arch = "wasm32", target_os = "unknown")))
            ))]
            impl EquationMetadataSource for $t {
                fn equation_metadata(&self) -> Option<&ValidatedModelMetadata> {
                    Some(self.metadata())
                }
            }
        )+
    };
}
impl_metadata_some!(
    pharmsol::dsl::RuntimeOdeModel,
    pharmsol::dsl::RuntimeAnalyticalModel,
    pharmsol::dsl::RuntimeSdeModel
);
