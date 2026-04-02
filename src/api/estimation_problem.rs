use anyhow::{bail, Result};
use pharmsol::{Data, Equation};
use serde::Serialize;

use crate::algorithms::Algorithm;
use crate::estimation::nonparametric::Prior;
use crate::model::ModelDefinition;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonparametricMethod {
    Npag(NpagOptions),
    Npbo(NpboOptions),
    Npcat(NpcatOptions),
    Npcma(NpcmaOptions),
    Npod(NpodOptions),
    Npopt(NpoptOptions),
    Nppso(NppsoOptions),
    Npsah(NpsahOptions),
    Npsah2(Npsah2Options),
    Nexus(NexusOptions),
    Npxo(NpxoOptions),
    Postprob(PostProbOptions),
}

impl NonparametricMethod {
    pub fn algorithm(self) -> Algorithm {
        match self {
            NonparametricMethod::Npag(_) => Algorithm::NPAG,
            NonparametricMethod::Npbo(_) => Algorithm::NPBO,
            NonparametricMethod::Npcat(_) => Algorithm::NPCAT,
            NonparametricMethod::Npcma(_) => Algorithm::NPCMA,
            NonparametricMethod::Npod(_) => Algorithm::NPOD,
            NonparametricMethod::Npopt(_) => Algorithm::NPOPT,
            NonparametricMethod::Nppso(_) => Algorithm::NPPSO,
            NonparametricMethod::Npsah(_) => Algorithm::NPSAH,
            NonparametricMethod::Npsah2(_) => Algorithm::NPSAH2,
            NonparametricMethod::Nexus(_) => Algorithm::NEXUS,
            NonparametricMethod::Npxo(_) => Algorithm::NPXO,
            NonparametricMethod::Postprob(_) => Algorithm::POSTPROB,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NpagOptions;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NpboOptions;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NpcatOptions;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NpcmaOptions;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NpodOptions;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NpoptOptions;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NppsoOptions;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NpsahOptions;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Npsah2Options;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NexusOptions;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NpxoOptions;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PostProbOptions;

#[derive(Debug, Clone, PartialEq, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LoggingLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl Default for LoggingLevel {
    fn default() -> Self {
        Self::Info
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
pub struct AlgorithmTuning {
    pub min_distance: f64,
    pub nm_steps: usize,
    pub tolerance: f64,
}

impl Default for AlgorithmTuning {
    fn default() -> Self {
        Self {
            min_distance: 1e-4,
            nm_steps: 100,
            tolerance: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
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
