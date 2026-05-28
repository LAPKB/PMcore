use anyhow::{anyhow, Result};
use pharmsol::{AssayErrorModel, AssayErrorModels, Data, Equation};

use crate::algorithms::Algorithm;
use crate::api::error_models::ErrorModels;

use crate::model::{EquationMetadataSource, Model, ModelBuilder, Parameter, ParameterSpace};
use crate::results::FitResult;

#[derive(Debug, Clone)]
pub struct EstimationProblem<E: Equation> {
    pub(crate) model: Model<E>,
    pub(crate) data: Data,
    pub(crate) error_models: ErrorModels,
    pub(crate) algorithm: Algorithm,
    pub(crate) parameters: ParameterSpace,
}

impl<E: Equation> EstimationProblem<E> {
    pub fn builder(equation: E, data: Data) -> EstimationProblemBuilder<E> {
        EstimationProblemBuilder {
            model: Model::builder(equation),
            data,
            parameters: ParameterSpace::new(),
            algorithm: None,
            error_models: Vec::new(),
        }
    }
}

impl<E: Equation + Clone + Send + 'static + EquationMetadataSource> EstimationProblem<E> {
    pub fn fit(self) -> Result<FitResult<E>> {
        unimplemented!("fit method is not implemented yet")
    }
}

// --- The Unified Builder ---

pub struct EstimationProblemBuilder<E: Equation> {
    model: ModelBuilder<E>,
    data: Data,
    parameters: ParameterSpace,
    algorithm: Option<Algorithm>,
    error_models: Vec<(String, AssayErrorModel)>,
}

impl<E: Equation> EstimationProblemBuilder<E> {
    pub fn algorithm(mut self, algorithm: impl Into<Algorithm>) -> Self {
        self.algorithm = Some(algorithm.into());
        self
    }

    /// Add a single parameter
    pub fn parameter(mut self, parameter: Parameter) -> Self {
        self.parameters.push(parameter);
        self
    }

    /// Batch add multiple parameters
    pub fn parameters(mut self, parameters: impl IntoIterator<Item = Parameter>) -> Self {
        for param in parameters {
            self.parameters.push(param);
        }
        self
    }

    /// Add an error model
    // TODO: result
    pub fn error(mut self, name: impl Into<String>, model: AssayErrorModel) -> Self {
        self.error_models.push((name.into(), model));
        self
    }

    // Helper for mapping the internal model builder if needed
    fn with_model_builder(mut self, map: impl FnOnce(ModelBuilder<E>) -> ModelBuilder<E>) -> Self {
        self.model = map(self.model);
        self
    }
}

impl<E: Equation + EquationMetadataSource> EstimationProblemBuilder<E> {
    /// Validates all parameters and constructs the problem.
    pub fn build(self) -> Result<EstimationProblem<E>> {
        let algorithm = self
            .algorithm
            .ok_or_else(|| anyhow!("an algorithm must be selected before building"))?;

        if self.parameters.is_empty() {
            anyhow::bail!("at least one parameter is required");
        }

        if self.error_models.is_empty() {
            anyhow::bail!("at least one error model is required");
        }

        let mut all_errors = AssayErrorModels::new();
        for (name, error_model) in self.error_models {
            let outeq = self
                .model
                .output_index(&name)
                .ok_or_else(|| anyhow!("unknown equation output label: {name}"))?;

            all_errors = all_errors.add(outeq, error_model)?;
        }

        let model = self.model.build()?;

        Ok(EstimationProblem {
            model,
            data: self.data,
            error_models: ErrorModels::Nonparametric(all_errors),
            algorithm,
            parameters: self.parameters,
        })
    }
}

impl<E: Equation + Clone + Send + 'static + EquationMetadataSource> EstimationProblemBuilder<E> {
    pub fn fit(self) -> Result<crate::results::FitResult<E>> {
        // Automatically builds and executes
        self.build()?.fit()
    }
}
