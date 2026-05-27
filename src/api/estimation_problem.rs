use anyhow::Result;
use pharmsol::{AssayErrorModel, AssayErrorModels, Data, Equation};

use crate::algorithms::Algorithm;
use crate::api::error_models::ErrorModels;

use crate::model::{
    CovariateSpec, EquationMetadataSource, ModelDefinition, ModelDefinitionBuilder, ModelMetadata,
    Parameter, VariabilityModel,
};
use crate::results::FitResult;

#[derive(Debug, Clone)]
pub struct EstimationProblem<E: Equation> {
    pub(crate) model: ModelDefinition<E>,
    pub(crate) data: Data,
    pub(crate) error_models: ErrorModels,
    pub(crate) algorithm: Algorithm,
}

impl<E: Equation> EstimationProblem<E> {
    pub fn builder(equation: E, data: Data) -> EstimationProblemBuilder<E> {
        EstimationProblemBuilder {
            model: ModelDefinition::builder(equation),
            data,
        }
    }
}

impl<E: Equation + Clone + Send + 'static + EquationMetadataSource> EstimationProblem<E> {
    pub fn fit(self) -> Result<FitResult<E>> {
        unimplemented!("fit method is not implemented yet")
    }

    pub fn fit_with_progress<F>(self, _on_progress: F) -> Result<crate::results::FitResult<E>>
    where
        F: FnMut(crate::api::FitProgress),
    {
        unimplemented!("fit method is not implemented yet")
    }
}

pub struct EstimationProblemBuilder<E: Equation> {
    model: ModelDefinitionBuilder<E>,
    data: Data,
}

impl<E: Equation> EstimationProblemBuilder<E> {
    /// Select the estimation algorithm. Accepts any algorithm configuration
    /// (e.g. [`Npag`](crate::algorithms::nonparametric::Npag)) that converts
    /// into [`Algorithm`].
    pub fn algorithm(
        self,
        algorithm: impl Into<Algorithm>,
    ) -> NonparametricEstimationProblemBuilder<E> {
        NonparametricEstimationProblemBuilder::new(self, algorithm.into())
    }
}

impl<E: Equation + EquationMetadataSource> EstimationProblemBuilder<E> {
    pub fn parameter(self, parameter: Parameter) -> Result<Self> {
        self.map_model_builder(|model| model.parameter(parameter))
    }

    pub fn variability(self, variability: VariabilityModel) -> Self {
        self.with_model_builder(|model| model.variability(variability))
    }

    pub fn covariates(self, covariates: CovariateSpec) -> Self {
        self.with_model_builder(|model| model.covariates(covariates))
    }

    pub fn metadata(self, metadata: ModelMetadata) -> Self {
        self.with_model_builder(|model| model.metadata(metadata))
    }

    fn map_model_builder(
        self,
        map: impl FnOnce(ModelDefinitionBuilder<E>) -> Result<ModelDefinitionBuilder<E>>,
    ) -> Result<Self> {
        let EstimationProblemBuilder { model, data } = self;

        Ok(Self {
            model: map(model)?,
            data,
        })
    }

    fn with_model_builder(
        self,
        map: impl FnOnce(ModelDefinitionBuilder<E>) -> ModelDefinitionBuilder<E>,
    ) -> Self {
        let EstimationProblemBuilder { model, data } = self;

        Self {
            model: map(model),
            data,
        }
    }
}

pub struct NonparametricEstimationProblemBuilder<E: Equation> {
    builder: EstimationProblemBuilder<E>,
    algorithm: Algorithm,
    error_models: Option<AssayErrorModels>,
}

impl<E: Equation> NonparametricEstimationProblemBuilder<E> {
    fn new(builder: EstimationProblemBuilder<E>, algorithm: Algorithm) -> Self {
        Self {
            builder,
            algorithm,
            error_models: None,
        }
    }

    fn with_builder(
        self,
        map: impl FnOnce(EstimationProblemBuilder<E>) -> EstimationProblemBuilder<E>,
    ) -> Self {
        let NonparametricEstimationProblemBuilder {
            builder,
            algorithm,
            error_models,
        } = self;

        Self {
            builder: map(builder),
            algorithm,
            error_models,
        }
    }
}

impl<E: Equation + EquationMetadataSource> NonparametricEstimationProblemBuilder<E> {
    pub fn parameter(self, parameter: Parameter) -> Result<Self> {
        self.map_builder(|builder| builder.parameter(parameter))
    }

    pub fn variability(self, variability: VariabilityModel) -> Self {
        self.with_builder(|builder| builder.variability(variability))
    }

    pub fn covariates(self, covariates: CovariateSpec) -> Self {
        self.with_builder(|builder| builder.covariates(covariates))
    }

    pub fn metadata(self, metadata: ModelMetadata) -> Self {
        self.with_builder(|builder| builder.metadata(metadata))
    }

    pub fn error(mut self, name: &str, model: AssayErrorModel) -> Result<Self> {
        let outeq = self
            .builder
            .model
            .output_index(name)
            .ok_or_else(|| anyhow::anyhow!("unknown equation output label: {name}"))?;

        self.error_models = Some(match self.error_models.take() {
            None => AssayErrorModels::new().add(outeq, model)?,
            Some(models) => models.add(outeq, model)?,
        });

        Ok(self)
    }

    pub fn build(self) -> Result<EstimationProblem<E>> {
        let model = self.builder.model.build()?;
        let error_models = self
            .error_models
            .ok_or_else(|| anyhow::anyhow!("error models are required"))?;

        Ok(EstimationProblem {
            model,
            data: self.builder.data,
            error_models: ErrorModels::Nonparametric(error_models),
            algorithm: self.algorithm,
        })
    }

    fn map_builder(
        self,
        map: impl FnOnce(EstimationProblemBuilder<E>) -> Result<EstimationProblemBuilder<E>>,
    ) -> Result<Self> {
        let NonparametricEstimationProblemBuilder {
            builder,
            algorithm,
            error_models,
        } = self;

        Ok(Self {
            builder: map(builder)?,
            algorithm,
            error_models,
        })
    }
}

impl<E: Equation + Clone + Send + 'static + EquationMetadataSource>
    NonparametricEstimationProblemBuilder<E>
{
    pub fn fit(self) -> Result<crate::results::FitResult<E>> {
        self.build()?.fit()
    }

    pub fn fit_with_progress<F>(self, on_progress: F) -> Result<crate::results::FitResult<E>>
    where
        F: FnMut(crate::api::FitProgress),
    {
        self.build()?.fit_with_progress(on_progress)
    }
}
