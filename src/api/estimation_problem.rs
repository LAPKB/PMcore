use anyhow::Result;
use pharmsol::{AssayErrorModel, AssayErrorModels, Data, Equation};
use serde::{Deserialize, Serialize};

use crate::algorithms::Algorithm;
use crate::api::error_models::ErrorModels;
use crate::api::SaemConfig;
use crate::estimation::nonparametric::Prior;
use crate::model::{
    CovariateSpec, EquationMetadataSource, ModelDefinition, ModelDefinitionBuilder, ModelMetadata,
    Parameter, VariabilityModel,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConvergenceOptions {
    pub likelihood: f64,
    pub pyl: f64,
    pub eps: f64,
}

impl Default for ConvergenceOptions {
    fn default() -> Self {
        Self {
            likelihood: 1e-4,
            pyl: 1e-2,
            eps: 1e-2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AlgorithmTuning {
    pub min_distance: f64,
    pub nm_steps: usize,
    pub tolerance: f64,
    pub saem: SaemConfig,
}

impl Default for AlgorithmTuning {
    fn default() -> Self {
        Self {
            min_distance: 1e-4,
            nm_steps: 100,
            tolerance: 1e-6,
            saem: SaemConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeOptions {
    pub cycles: usize,
    pub cache: bool,
    pub progress: bool,
    pub idelta: f64,
    pub tad: f64,
    pub prior: Option<Prior>,
    pub convergence: ConvergenceOptions,
    pub tuning: AlgorithmTuning,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            cycles: 100,
            cache: true,
            progress: true,
            idelta: 0.12,
            tad: 0.0,
            prior: None,
            convergence: ConvergenceOptions::default(),
            tuning: AlgorithmTuning::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EstimationProblem<E: Equation> {
    pub(crate) model: ModelDefinition<E>,
    pub(crate) data: Data,
    pub(crate) error_models: ErrorModels,
    pub(crate) algorithm: Algorithm,
    pub(crate) runtime: RuntimeOptions,
}

impl<E: Equation> EstimationProblem<E> {
    pub fn builder(equation: E, data: Data) -> EstimationProblemBuilder<E> {
        EstimationProblemBuilder {
            model: ModelDefinition::builder(equation),
            data,
            runtime: Some(RuntimeOptions::default()),
        }
    }
}

impl<E: Equation + Clone + Send + 'static + EquationMetadataSource> EstimationProblem<E> {
    pub fn fit(self) -> Result<crate::results::FitResult<E>> {
        crate::api::fit(self)
    }

    pub fn fit_with_progress<F>(self, on_progress: F) -> Result<crate::results::FitResult<E>>
    where
        F: FnMut(crate::api::FitProgress),
    {
        crate::api::fit_with_progress(self, on_progress)
    }
}

pub struct EstimationProblemBuilder<E: Equation> {
    model: ModelDefinitionBuilder<E>,
    data: Data,
    runtime: Option<RuntimeOptions>,
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

    pub fn cycles(self, cycles: usize) -> Self {
        self.with_runtime_options(|runtime| runtime.cycles = cycles)
    }

    pub fn cache(self, enabled: bool) -> Self {
        self.with_runtime_options(|runtime| runtime.cache = enabled)
    }

    pub fn progress(self, enabled: bool) -> Self {
        self.with_runtime_options(|runtime| runtime.progress = enabled)
    }

    pub fn idelta(self, value: f64) -> Self {
        self.with_runtime_options(|runtime| runtime.idelta = value)
    }

    pub fn tad(self, value: f64) -> Self {
        self.with_runtime_options(|runtime| runtime.tad = value)
    }

    pub fn prior(self, prior: Prior) -> Self {
        self.with_runtime_options(|runtime| runtime.prior = Some(prior))
    }

    pub fn convergence(self, convergence: ConvergenceOptions) -> Self {
        self.with_runtime_options(|runtime| runtime.convergence = convergence)
    }

    pub fn tuning(self, tuning: AlgorithmTuning) -> Self {
        self.with_runtime_options(|runtime| runtime.tuning = tuning)
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
        let EstimationProblemBuilder {
            model,
            data,

            runtime,
        } = self;

        Ok(Self {
            model: map(model)?,
            data,

            runtime,
        })
    }

    fn with_model_builder(
        self,
        map: impl FnOnce(ModelDefinitionBuilder<E>) -> ModelDefinitionBuilder<E>,
    ) -> Self {
        let EstimationProblemBuilder {
            model,
            data,

            runtime,
        } = self;

        Self {
            model: map(model),
            data,

            runtime,
        }
    }
}

impl<E: Equation> EstimationProblemBuilder<E> {
    fn with_runtime_options(mut self, map: impl FnOnce(&mut RuntimeOptions)) -> Self {
        map(self.runtime.get_or_insert_with(RuntimeOptions::default));
        self
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

    pub fn cache(self, enabled: bool) -> Self {
        self.with_builder(|builder| builder.cache(enabled))
    }

    pub fn progress(self, enabled: bool) -> Self {
        self.with_builder(|builder| builder.progress(enabled))
    }

    pub fn idelta(self, value: f64) -> Self {
        self.with_builder(|builder| builder.idelta(value))
    }

    pub fn tad(self, value: f64) -> Self {
        self.with_builder(|builder| builder.tad(value))
    }

    pub fn prior(self, prior: Prior) -> Self {
        self.with_builder(|builder| builder.prior(prior))
    }

    pub fn convergence(self, convergence: ConvergenceOptions) -> Self {
        self.with_builder(|builder| builder.convergence(convergence))
    }

    pub fn tuning(self, tuning: AlgorithmTuning) -> Self {
        self.with_builder(|builder| builder.tuning(tuning))
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
            runtime: self.builder.runtime.unwrap_or_default(),
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
