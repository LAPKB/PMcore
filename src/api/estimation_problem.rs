use anyhow::{bail, Result};
use pharmsol::{Data, Equation};
use serde::{Deserialize, Serialize};

use crate::algorithms::Algorithm;
use crate::api::SaemConfig;
use crate::estimation::nonparametric::Prior;
use crate::model::ModelDefinition;

// =============================================================================
// Estimation method selection
// =============================================================================

/// Estimation method family and algorithm.
///
/// Serializes to `{"family": "nonparametric", "algorithm": "npag"}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimationMethod {
    Nonparametric(NonparametricMethod),
}

impl EstimationMethod {
    pub fn algorithm(self) -> Algorithm {
        match self {
            EstimationMethod::Nonparametric(method) => method.algorithm(),
        }
    }
}

impl Serialize for EstimationMethod {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct Wire<'a> {
            family: &'a str,
            algorithm: &'a str,
        }
        let (family, algorithm) = match self {
            EstimationMethod::Nonparametric(np) => ("nonparametric", np.name()),
        };
        Wire { family, algorithm }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EstimationMethod {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            family: String,
            algorithm: String,
        }
        let wire = Wire::deserialize(deserializer)?;
        match wire.family.to_lowercase().as_str() {
            "nonparametric" => NonparametricMethod::from_name(&wire.algorithm)
                .map(EstimationMethod::Nonparametric)
                .ok_or_else(|| {
                    serde::de::Error::custom(format!(
                        "unknown nonparametric algorithm: {}",
                        wire.algorithm
                    ))
                }),
            "parametric" => Err(serde::de::Error::custom(
                "parametric methods are not available in unified-platform-structure",
            )),
            other => Err(serde::de::Error::custom(format!(
                "unknown method family: {}",
                other
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonparametricMethod {
    Npag(NpagOptions),
    Npod(NpodOptions),
    Postprob(PostProbOptions),
}

impl NonparametricMethod {
    pub fn algorithm(self) -> Algorithm {
        match self {
            NonparametricMethod::Npag(_) => Algorithm::NPAG,
            NonparametricMethod::Npod(_) => Algorithm::NPOD,
            NonparametricMethod::Postprob(_) => Algorithm::POSTPROB,
        }
    }

    /// Wire name for this algorithm (lowercase).
    pub fn name(&self) -> &'static str {
        match self {
            NonparametricMethod::Npag(_) => "npag",
            NonparametricMethod::Npod(_) => "npod",
            NonparametricMethod::Postprob(_) => "postprob",
        }
    }

    /// Construct from a wire name (case-insensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "npag" => Some(NonparametricMethod::Npag(NpagOptions)),
            "npod" => Some(NonparametricMethod::Npod(NpodOptions)),
            "postprob" => Some(NonparametricMethod::Postprob(PostProbOptions)),
            _ => None,
        }
    }
}

// =============================================================================
// Algorithm option marker types
// =============================================================================

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NpagOptions;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NpodOptions;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostProbOptions;

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
        Self {
            write: true,
            path: None,
        }
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
    pub model: ModelDefinition<E>,
    pub data: Data,
    pub method: EstimationMethod,
    pub output: OutputPlan,
    pub runtime: RuntimeOptions,
}

impl<E: Equation> EstimationProblem<E> {
    pub fn builder(model: ModelDefinition<E>, data: Data) -> EstimationProblemBuilder<E> {
        EstimationProblemBuilder {
            model,
            data,
            method: None,
            output: Some(OutputPlan::default()),
            runtime: Some(RuntimeOptions::default()),
        }
    }
}

impl<E: Equation + Clone + Send + 'static> EstimationProblem<E> {
    pub fn run(self) -> Result<crate::results::FitResult<E>> {
        crate::api::fit(self)
    }

    pub fn run_with_progress<F>(self, on_progress: F) -> Result<crate::results::FitResult<E>>
    where
        F: FnMut(crate::api::FitProgress),
    {
        crate::api::fit_with_progress(self, on_progress)
    }
}

pub struct EstimationProblemBuilder<E: Equation> {
    model: ModelDefinition<E>,
    data: Data,
    method: Option<EstimationMethod>,
    output: Option<OutputPlan>,
    runtime: Option<RuntimeOptions>,
}

impl<E: Equation> EstimationProblemBuilder<E> {
    pub fn method(mut self, method: EstimationMethod) -> Self {
        self.method = Some(method);
        self
    }

    pub fn output(mut self, output: OutputPlan) -> Self {
        self.output = Some(output);
        self
    }

    pub fn runtime(mut self, runtime: RuntimeOptions) -> Self {
        self.runtime = Some(runtime);
        self
    }

    pub fn build(self) -> Result<EstimationProblem<E>> {
        let method = self
            .method
            .ok_or_else(|| anyhow::anyhow!("estimation method is required"))?;
        if self.model.parameters.is_empty() {
            bail!("estimation problem requires at least one parameter");
        }

        Ok(EstimationProblem {
            model: self.model,
            data: self.data,
            method,
            output: self.output.unwrap_or_default(),
            runtime: self.runtime.unwrap_or_default(),
        })
    }
}

impl<E: Equation + Clone + Send + 'static> EstimationProblemBuilder<E> {
    pub fn run(self) -> Result<crate::results::FitResult<E>> {
        self.build()?.run()
    }
}
