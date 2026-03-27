//! Parametric algorithm trait definition
//!
//! This module defines the [`ParametricAlgorithm`] trait that all parametric
//! population estimation algorithms must implement.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use pharmsol::{Data, Equation, ResidualErrorModels};

use crate::api::{
    EstimationMethod, EstimationProblem, OutputPlan, ParametricMethod, RuntimeOptions,
    SaemConfig,
};
use crate::compile::{CompiledProblem, StructuredCovariateDesign};
use crate::estimation::parametric::{
    IndividualEstimates, ParameterTransform as AlgorithmParameterTransform, ParametricWorkspace,
    Population, SufficientStats,
};
use crate::model::{CovariateSpec, ParameterDomain, ParameterSpace};
use crate::output::shared::RunConfiguration;

use super::super::Status;

#[derive(Debug, Clone)]
pub(crate) struct ParametricAlgorithmInput<E: Equation> {
    pub method: ParametricMethod,
    pub equation: E,
    pub data: Data,
    pub parameter_space: ParameterSpace,
    pub covariates: CovariateSpec,
    pub structured_covariates: StructuredCovariateDesign,
    pub residual_error_models: ResidualErrorModels,
    pub output: OutputPlan,
    pub runtime: RuntimeOptions,
}

impl<E: Equation> ParametricAlgorithmInput<E> {
    pub(crate) fn from_compiled_problem(problem: CompiledProblem<E>) -> Result<Self> {
        let method = match problem.method() {
            EstimationMethod::Parametric(method) => method,
            other => anyhow::bail!("parametric dispatcher received non-parametric method: {:?}", other),
        };

        let output = problem.output_plan().clone();
        let runtime = problem.runtime_options().clone();
        let covariates = problem.model.covariates.clone();
        let structured_covariates = problem.design.structured_covariates.clone();
        let (model, data) = problem.into_parts();

        let residual_error_models = model
            .observations
            .residual_error_models
            .clone()
            .ok_or_else(|| anyhow::anyhow!("parametric algorithms require residual_error_models"))?;

        Ok(Self {
            method,
            equation: model.equation,
            data,
            parameter_space: model.parameters,
            covariates,
            structured_covariates,
            residual_error_models,
            output,
            runtime,
        })
    }

    pub(crate) fn algorithm(&self) -> crate::algorithms::Algorithm {
        self.method.algorithm()
    }

    pub(crate) fn run_configuration(&self) -> RunConfiguration {
        RunConfiguration::new(
            self.algorithm(),
            &self.output,
            &self.runtime,
            self.parameter_space
                .iter()
                .map(|parameter| parameter.name.clone())
                .collect(),
        )
    }

    pub(crate) fn saem_config(&self) -> &SaemConfig {
        &self.runtime.tuning.saem
    }

    pub(crate) fn initial_population(&self) -> Result<Population> {
        Population::from_parameter_space(self.parameter_space.clone())
    }

    pub(crate) fn parameter_transforms(&self) -> Vec<AlgorithmParameterTransform> {
        self.parameter_space.iter().map(to_parameter_transform).collect()
    }
}

/// Configuration specific to parametric algorithms
#[derive(Debug, Clone)]
pub struct ParametricConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Number of burn-in iterations (for SAEM)
    pub burn_in: usize,
    /// Number of MCMC chains per subject
    pub n_chains: usize,
    /// Number of samples per chain per iteration
    pub n_samples: usize,
    /// Convergence tolerance for parameters
    pub parameter_tolerance: f64,
    /// Convergence tolerance for objective function
    pub objective_tolerance: f64,
    /// Whether to use simulated annealing in SAEM
    pub use_annealing: bool,
    /// Initial temperature for simulated annealing
    pub initial_temperature: f64,
}

impl Default for ParametricConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            burn_in: 200,
            n_chains: 1,
            n_samples: 1,
            parameter_tolerance: 1e-4,
            objective_tolerance: 1e-4,
            use_annealing: false,
            initial_temperature: 1.0,
        }
    }
}

/// Trait defining the interface for parametric population algorithms
///
/// This trait provides the common structure for algorithms that estimate
/// population parameters assuming a continuous (typically multivariate normal)
/// distribution for the random effects.
///
/// # Algorithm Workflow
///
/// 1. **Initialize**: Set up initial population parameters
/// 2. **E-step**: Compute or sample from conditional distribution p(η|y,θ)
/// 3. **M-step**: Update population parameters from E-step results
/// 4. **Evaluate**: Check convergence criteria
/// 5. **Repeat** until convergence or max iterations
///
/// # Type Parameters
///
/// * `E` - The equation type implementing pharmacokinetic/pharmacodynamic model
pub trait ParametricAlgorithm<E: Equation + Send + 'static>: Sync + Send {
    /// Get the equation/model
    fn equation(&self) -> &E;

    /// Get the data
    fn data(&self) -> &Data;

    /// Get the current population parameters
    fn population(&self) -> &Population;

    /// Get a mutable reference to population parameters
    fn population_mut(&mut self) -> &mut Population;

    /// Get the current individual estimates
    fn individual_estimates(&self) -> &IndividualEstimates;

    /// Get the current iteration number
    fn iteration(&self) -> usize;

    /// Increment the iteration counter and return new value
    fn increment_iteration(&mut self) -> usize;

    /// Get the current objective function value (-2LL)
    fn objective_function(&self) -> f64;

    /// Get the current algorithm status
    fn status(&self) -> &Status;

    /// Set the algorithm status
    fn set_status(&mut self, status: Status);

    // ========== Algorithm Steps ==========

    /// Initialize the algorithm
    ///
    /// Sets up initial population parameters, prepares data structures,
    /// and performs any pre-processing required before the main loop.
    fn initialize(&mut self) -> Result<()> {
        // Remove stop file if it exists
        if Path::new("stop").exists() {
            tracing::info!("Removing existing stop file prior to run");
            fs::remove_file("stop").context("Unable to remove previous stop file")?;
        }
        self.set_status(Status::Continue);
        Ok(())
    }

    /// Perform the E-step (Expectation step)
    ///
    /// This step computes or samples from the conditional distribution of
    /// individual parameters given the observations and current population parameters.
    ///
    /// - **FOCEI**: Finds the MAP estimate (mode) with a local curvature approximation
    /// - **SAEM**: Samples from p(η|y,θ) using MCMC
    fn e_step(&mut self) -> Result<()>;

    /// Perform the M-step (Maximization step)
    ///
    /// Updates the population parameters (μ, Ω) based on the E-step results.
    ///
    /// - **FOCEI**: Uses the subject modes and local curvature information
    /// - **SAEM**: Uses sufficient statistics from MCMC samples
    fn m_step(&mut self) -> Result<()>;

    /// Evaluate convergence and update status
    ///
    /// Checks various convergence criteria and determines whether to continue
    /// or stop the algorithm.
    fn evaluate(&mut self) -> Result<Status>;

    /// Log the current iteration state
    fn log_iteration(&mut self);

    /// Perform a single iteration of the algorithm
    ///
    /// Default implementation calls E-step, M-step, logging, and evaluation.
    fn next_iteration(&mut self) -> Result<Status> {
        let iter = self.increment_iteration();

        let span = tracing::info_span!("", "{}", format!("Iteration {}", iter));
        let _enter = span.enter();

        self.e_step()?;
        self.m_step()?;
        self.log_iteration();
        self.evaluate()
    }

    /// Run the full estimation procedure
    ///
    /// Initializes the algorithm and iterates until convergence or stopping criteria.
    fn fit(&mut self) -> Result<ParametricWorkspace<E>> {
        self.initialize()?;

        loop {
            match self.next_iteration()? {
                Status::Continue => continue,
                Status::Stop(_) => break,
            }
        }

        self.into_result()
    }

    /// Convert the algorithm state into a result object
    fn into_result(&self) -> Result<ParametricWorkspace<E>>;

    // ========== Optional Methods ==========

    /// Get sufficient statistics (for SAEM-like algorithms)
    fn sufficient_stats(&self) -> Option<&SufficientStats> {
        None
    }

    /// Perform optimization of error model parameters
    ///
    /// Some algorithms may optimize error model parameters alongside population parameters.
    fn optimize_error_model(&mut self) -> Result<()> {
        // Default: no optimization
        Ok(())
    }

    /// Apply constraints to population parameters
    ///
    /// Ensures parameters stay within bounds and covariance matrix remains positive definite.
    fn apply_constraints(&mut self) -> Result<()> {
        // Default: no additional constraints
        Ok(())
    }
}

/// Dispatch function for parametric algorithms
///
/// Creates the appropriate algorithm instance based on settings.
pub fn dispatch_parametric_algorithm<E: Equation + Clone + Send + 'static>(
    problem: EstimationProblem<E>,
) -> Result<Box<dyn ParametricAlgorithm<E>>> {
    let compiled = problem.compile()?;
    let input = ParametricAlgorithmInput::from_compiled_problem(compiled)?;

    dispatch_parametric_algorithm_input(input)
}

pub(crate) fn dispatch_parametric_algorithm_input<E: Equation + Clone + Send + 'static>(
    input: ParametricAlgorithmInput<E>,
) -> Result<Box<dyn ParametricAlgorithm<E>>> {
    use super::focei::FoceiAlgorithm;
    use super::saem::FSAEM;

    match input.method {
        ParametricMethod::Saem(_) => {
            let saem = FSAEM::create(input)?;
            Ok(saem as Box<dyn ParametricAlgorithm<E>>)
        }
        ParametricMethod::Focei(_) => {
            let focei = FoceiAlgorithm::create(input)?;
            Ok(focei as Box<dyn ParametricAlgorithm<E>>)
        }
        ParametricMethod::It2b(_) => {
            // TODO: Implement IT2B
            anyhow::bail!("IT2B algorithm not yet implemented")
        }
    }
}

pub(crate) fn run_parametric_algorithm<E: Equation + Clone + Send + 'static>(
    input: ParametricAlgorithmInput<E>,
) -> Result<ParametricWorkspace<E>> {
    let mut algorithm = dispatch_parametric_algorithm_input(input)?;
    algorithm.fit()
}

fn to_parameter_transform(
    parameter: &crate::model::ParameterSpec,
) -> AlgorithmParameterTransform {
    match parameter.transform {
        crate::model::ParameterTransform::Identity => AlgorithmParameterTransform::None,
        crate::model::ParameterTransform::LogNormal => AlgorithmParameterTransform::LogNormal,
        crate::model::ParameterTransform::Logit => {
            let (lower, upper) = bounded_domain(parameter);
            AlgorithmParameterTransform::Logit { lower, upper }
        }
        crate::model::ParameterTransform::Probit => {
            let (lower, upper) = bounded_domain(parameter);
            AlgorithmParameterTransform::Probit { lower, upper }
        }
    }
}

fn bounded_domain(parameter: &crate::model::ParameterSpec) -> (f64, f64) {
    match parameter.domain {
        ParameterDomain::Bounded { lower, upper } => (lower, upper),
        ParameterDomain::Positive { lower, upper } => {
            (lower.unwrap_or(0.0), upper.unwrap_or(1.0))
        }
        ParameterDomain::Unbounded { lower, upper } => {
            (lower.unwrap_or(0.0), upper.unwrap_or(1.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ParametricAlgorithmInput;
    use anyhow::Result;
    use pharmsol::{AssayErrorModel, ErrorPoly, ResidualErrorModel, ResidualErrorModels, Subject};

    use crate::api::{EstimationMethod, EstimationProblem, ParametricMethod, SaemOptions};
    use crate::model::{
        CovariateEffectsSpec, CovariateSpec, ModelDefinition, ObservationChannel,
        ObservationSpec, ParameterSpace, ParameterSpec,
    };
    use crate::prelude::*;

    fn equation() -> equation::ODE {
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke);
                dx[0] = -ke * x[0] + b[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, v);
                y[0] = x[0] / v;
            },
        )
    }

    #[test]
    fn compiled_parametric_input_preserves_structured_covariates() -> Result<()> {
        let data = pharmsol::Data::new(vec![
            Subject::builder("1")
                .covariate("wt", 0.0, 70.0)
                .bolus(0.0, 100.0, 0)
                .observation(1.0, 10.0, 0)
                .build(),
        ]);
        let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
        let residual_error =
            ResidualErrorModels::new().add(0, ResidualErrorModel::combined(0.5, 0.1));
        let observations = ObservationSpec::new()
            .add_channel(ObservationChannel::continuous(0, "cp"))
            .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?)
            .with_residual_error_models(residual_error);

        let model = ModelDefinition::builder(equation())
            .parameters(
                ParameterSpace::new()
                    .add(ParameterSpec::positive("ke"))
                    .add(ParameterSpec::positive("v")),
            )
            .observations(observations)
            .covariates(CovariateSpec::Structured(CovariateEffectsSpec {
                subject_effects: Some(CovariateModel::new(
                    vec!["ke", "v"],
                    vec!["wt"],
                    vec![vec![true], vec![false]],
                )?),
                occasion_effects: None,
            }))
            .build()?;

        let compiled = EstimationProblem::builder(model, data)
            .method(EstimationMethod::Parametric(ParametricMethod::Saem(SaemOptions)))
            .build()?
            .compile()?;

        let input = ParametricAlgorithmInput::from_compiled_problem(compiled)?;

        assert!(matches!(input.covariates, CovariateSpec::Structured(_)));
        assert_eq!(input.structured_covariates.subject_columns, vec!["wt"]);
        assert_eq!(input.structured_covariates.subject_rows.len(), 1);
        assert_eq!(input.structured_covariates.subject_rows[0].values, vec![Some(70.0)]);
        Ok(())
    }
}
