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

// =============================================================================
// Method selection
// =============================================================================

pub trait MethodSpec {
    type Builder<E: Equation>;

    fn attach<E: Equation>(self, builder: EstimationProblemBuilder<E>) -> Self::Builder<E>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Npag;

impl Npag {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Npod;

impl Npod {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostProb;

impl PostProb {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum NonparametricMethod {
    Npag(Npag),
    Npod(Npod),
    PostProb(PostProb),
}

impl NonparametricMethod {
    pub fn algorithm(self) -> Algorithm {
        match self {
            Self::Npag(_) => Algorithm::NPAG,
            Self::Npod(_) => Algorithm::NPOD,
            Self::PostProb(_) => Algorithm::POSTPROB,
        }
    }
}

impl MethodSpec for Npag {
    type Builder<E: Equation> = NonparametricEstimationProblemBuilder<E>;

    fn attach<E: Equation>(self, builder: EstimationProblemBuilder<E>) -> Self::Builder<E> {
        NonparametricEstimationProblemBuilder::new(builder, NonparametricMethod::Npag(self))
    }
}

impl MethodSpec for Npod {
    type Builder<E: Equation> = NonparametricEstimationProblemBuilder<E>;

    fn attach<E: Equation>(self, builder: EstimationProblemBuilder<E>) -> Self::Builder<E> {
        NonparametricEstimationProblemBuilder::new(builder, NonparametricMethod::Npod(self))
    }
}

impl MethodSpec for PostProb {
    type Builder<E: Equation> = NonparametricEstimationProblemBuilder<E>;

    fn attach<E: Equation>(self, builder: EstimationProblemBuilder<E>) -> Self::Builder<E> {
        NonparametricEstimationProblemBuilder::new(builder, NonparametricMethod::PostProb(self))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputPlan {
    pub write: bool,
    pub path: Option<String>,
}

impl OutputPlan {
    pub fn disabled() -> Self {
        Self {
            write: false,
            path: None,
        }
    }
}

impl Default for OutputPlan {
    fn default() -> Self {
        Self::disabled()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LoggingLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingOptions {
    pub initialize: bool,
    pub level: LoggingLevel,
    pub write: bool,
    pub stdout: bool,
}

impl Default for LoggingOptions {
    fn default() -> Self {
        Self {
            initialize: false,
            level: LoggingLevel::Info,
            write: false,
            stdout: true,
        }
    }
}

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
    pub logging: LoggingOptions,
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
            logging: LoggingOptions::default(),
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
    pub(crate) method: NonparametricMethod,
    pub(crate) output: OutputPlan,
    pub(crate) runtime: RuntimeOptions,
}

impl<E: Equation> EstimationProblem<E> {
    pub fn builder(equation: E, data: Data) -> EstimationProblemBuilder<E> {
        EstimationProblemBuilder {
            model: ModelDefinition::builder(equation),
            data,
            output: Some(OutputPlan::default()),
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
    output: Option<OutputPlan>,
    runtime: Option<RuntimeOptions>,
}

impl<E: Equation> EstimationProblemBuilder<E> {
    pub fn method<M: MethodSpec>(self, method: M) -> M::Builder<E> {
        method.attach(self)
    }

    pub fn output_dir(self, path: impl Into<String>) -> Self {
        self.with_output_plan(|output| {
            output.write = true;
            output.path = Some(path.into());
        })
    }

    pub fn no_output(self) -> Self {
        self.with_output_plan(|output| {
            output.write = false;
            output.path = None;
        })
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

    pub fn initialize_logs(self) -> Self {
        self.with_runtime_options(|runtime| runtime.logging.initialize = true)
    }

    pub fn log_level(self, level: LoggingLevel) -> Self {
        self.with_runtime_options(|runtime| runtime.logging.level = level)
    }

    pub fn write_logs(self, enabled: bool) -> Self {
        self.with_runtime_options(|runtime| runtime.logging.write = enabled)
    }

    pub fn stdout_logs(self, enabled: bool) -> Self {
        self.with_runtime_options(|runtime| runtime.logging.stdout = enabled)
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
            output,
            runtime,
        } = self;

        Ok(Self {
            model: map(model)?,
            data,
            output,
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
            output,
            runtime,
        } = self;

        Self {
            model: map(model),
            data,
            output,
            runtime,
        }
    }
}

impl<E: Equation> EstimationProblemBuilder<E> {
    fn with_output_plan(mut self, map: impl FnOnce(&mut OutputPlan)) -> Self {
        map(self.output.get_or_insert_with(OutputPlan::default));
        self
    }

    fn with_runtime_options(mut self, map: impl FnOnce(&mut RuntimeOptions)) -> Self {
        map(self.runtime.get_or_insert_with(RuntimeOptions::default));
        self
    }
}

pub struct NonparametricEstimationProblemBuilder<E: Equation> {
    builder: EstimationProblemBuilder<E>,
    method: NonparametricMethod,
    error_models: Option<AssayErrorModels>,
}

impl<E: Equation> NonparametricEstimationProblemBuilder<E> {
    fn new(builder: EstimationProblemBuilder<E>, method: NonparametricMethod) -> Self {
        Self {
            builder,
            method,
            error_models: None,
        }
    }

    pub fn output_dir(self, path: impl Into<String>) -> Self {
        self.with_builder(|builder| builder.output_dir(path))
    }

    pub fn no_output(self) -> Self {
        self.with_builder(|builder| builder.no_output())
    }

    pub fn cycles(self, cycles: usize) -> Self {
        self.with_builder(|builder| builder.cycles(cycles))
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

    pub fn initialize_logs(self) -> Self {
        self.with_builder(|builder| builder.initialize_logs())
    }

    pub fn log_level(self, level: LoggingLevel) -> Self {
        self.with_builder(|builder| builder.log_level(level))
    }

    pub fn write_logs(self, enabled: bool) -> Self {
        self.with_builder(|builder| builder.write_logs(enabled))
    }

    pub fn stdout_logs(self, enabled: bool) -> Self {
        self.with_builder(|builder| builder.stdout_logs(enabled))
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
            method,
            error_models,
        } = self;

        Self {
            builder: map(builder),
            method,
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
            method: self.method,
            output: self.builder.output.unwrap_or_default(),
            runtime: self.builder.runtime.unwrap_or_default(),
        })
    }

    fn map_builder(
        self,
        map: impl FnOnce(EstimationProblemBuilder<E>) -> Result<EstimationProblemBuilder<E>>,
    ) -> Result<Self> {
        let NonparametricEstimationProblemBuilder {
            builder,
            method,
            error_models,
        } = self;

        Ok(Self {
            builder: map(builder)?,
            method,
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
