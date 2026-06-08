use anyhow::{anyhow, Result};
use pharmsol::{
    AssayErrorModel, AssayErrorModels, Data, Equation, ResidualErrorModel, ResidualErrorModels,
};

use crate::model::parameter_space::{BoundedParameter, ParameterSpace, UnboundedParameter};
use crate::model::{EquationMetadataSource, Model, ModelBuilder};

pub trait Framework {
    type ErrorModels;
    type Parameters;
}
pub struct Parametric;
impl Framework for Parametric {
    type ErrorModels = ResidualErrorModels;
    type Parameters = ParameterSpace<UnboundedParameter>;
}
pub struct NonParametric;

impl Framework for NonParametric {
    type ErrorModels = AssayErrorModels;
    type Parameters = ParameterSpace<BoundedParameter>;
}

#[derive(Debug, Clone)]
pub struct EstimationProblem<E: Equation, F: Framework> {
    pub(crate) model: Model<E>,
    pub(crate) data: Data,
    pub(crate) error_models: F::ErrorModels,
    pub(crate) parameters: F::Parameters,
}

pub struct EstimationProblemBuilder<E: Equation> {
    model: ModelBuilder<E>,
    data: Data,
}

impl<E: Equation> EstimationProblemBuilder<E> {
    pub fn nonparametric(self) -> NonParametricBuilder<E> {
        NonParametricBuilder {
            model: self.model,
            data: self.data,
            parameters: ParameterSpace::<BoundedParameter>::new(),
            error_models: Vec::new(),
        }
    }

    pub fn parametric(self) -> ParametricBuilder<E> {
        ParametricBuilder {
            model: self.model,
            data: self.data,
            parameters: ParameterSpace::<UnboundedParameter>::new(),
            error_models: Vec::new(),
        }
    }
}

impl<E: Equation> EstimationProblem<E, NonParametric> {
    pub fn builder(equation: E, data: Data) -> EstimationProblemBuilder<E> {
        EstimationProblemBuilder {
            model: Model::builder(equation),
            data,
        }
    }
}

pub struct NonParametricBuilder<E: Equation> {
    model: ModelBuilder<E>,
    data: Data,
    parameters: ParameterSpace<BoundedParameter>,
    error_models: Vec<(String, AssayErrorModel)>,
}

impl<E: Equation> NonParametricBuilder<E> {
    pub fn parameter(mut self, parameter: impl Into<BoundedParameter>) -> Self {
        self.parameters.push(parameter.into());
        self
    }

    pub fn parameters<P, I>(mut self, parameters: I) -> Self
    where
        P: Into<BoundedParameter>,
        I: IntoIterator<Item = P>,
    {
        for param in parameters {
            self.parameters.push(param.into());
        }
        self
    }

    pub fn error(mut self, name: impl Into<String>, model: AssayErrorModel) -> Self {
        self.error_models.push((name.into(), model));
        self
    }
}

impl<E: Equation + EquationMetadataSource> NonParametricBuilder<E> {
    pub fn build(self) -> Result<EstimationProblem<E, NonParametric>> {
        if self.parameters.is_empty() {
            anyhow::bail!("at least one parameter is required for non-parametric models");
        }

        if self.error_models.is_empty() {
            anyhow::bail!("at least one assay error model is required");
        }

        let mut all_errors = AssayErrorModels::new();
        for (name, error_model) in self.error_models {
            let outeq = self
                .model
                .output_index(&name)
                .ok_or_else(|| anyhow!("unknown equation output label: {name}"))?;

            all_errors = all_errors.add(outeq, error_model)?;
        }

        Ok(EstimationProblem {
            model: self.model.build()?,
            data: self.data,
            error_models: all_errors,
            parameters: self.parameters,
        })
    }
}

pub struct ParametricBuilder<E: Equation> {
    model: ModelBuilder<E>,
    data: Data,
    parameters: ParameterSpace<UnboundedParameter>,
    error_models: Vec<(String, ResidualErrorModel)>,
}

impl<E: Equation> ParametricBuilder<E> {
    pub fn parameter(mut self, parameter: impl Into<UnboundedParameter>) -> Self {
        self.parameters.push(parameter.into());
        self
    }

    pub fn parameters<P, I>(mut self, parameters: I) -> Self
    where
        P: Into<UnboundedParameter>,
        I: IntoIterator<Item = P>,
    {
        for param in parameters {
            self.parameters.push(param.into());
        }
        self
    }

    pub fn error(mut self, name: impl Into<String>, model: ResidualErrorModel) -> Self {
        self.error_models.push((name.into(), model));
        self
    }
}

impl<E: Equation + EquationMetadataSource> ParametricBuilder<E> {
    pub fn build(self) -> Result<EstimationProblem<E, Parametric>> {
        if self.parameters.is_empty() {
            anyhow::bail!("at least one parameter is required for parametric models");
        }

        if self.error_models.is_empty() {
            anyhow::bail!("at least one residual error model is required");
        }

        let mut all_errors = ResidualErrorModels::new();
        for (name, error_model) in self.error_models {
            let outeq = self
                .model
                .output_index(&name)
                .ok_or_else(|| anyhow!("unknown equation output label: {name}"))?;

            all_errors = all_errors.add(outeq, error_model);
        }

        Ok(EstimationProblem {
            model: self.model.build()?,
            data: self.data,
            error_models: all_errors, // Strongly typed as ResidualErrorModels!
            parameters: self.parameters,
        })
    }
}
